import torch
from recstudio.data.dataset import AEDataset
from recstudio.model.basemodel import BaseRetriever, Recommender
from recstudio.model.loss_func import SoftmaxLoss
from recstudio.model.module import MLPModule
from recstudio.model.scorer import InnerProductScorer
from recstudio.model.ae.diffusion_utils.script_util import create_gaussian_diffusion
from recstudio.model.ae.diffusion_utils.nn import ScoreNet
from recstudio.model.ae.diffusion_utils.resample import create_named_schedule_sampler, LossAwareSampler


class DDPMVAEQueryEncoder(torch.nn.Module):
    def __init__(self, fiid, num_items, embed_dim, dropout_rate, item_encoder, share_item_encoder, diffusion_steps, learn_sigma, 
                sigma_small, noise_schedule, use_kl, predict_xstart, rescale_timesteps, rescale_learned_sigmas, timestep_respacing, schedule_sampler, kl_lambda,
                clip_denoised, use_ddim):
        super().__init__()

        self.fiid = fiid
        self.embed_dim = embed_dim
        self.kl_lambda = kl_lambda
        self.clip_denoised = clip_denoised
        self.use_ddim = use_ddim

        self.diffusion_model = ScoreNet(embed_dim, dropout_rate, learn_sigma)
        self.diffusion = create_gaussian_diffusion(steps=diffusion_steps, learn_sigma=learn_sigma, sigma_small=sigma_small, noise_schedule=noise_schedule,
                                                   use_kl=use_kl, predict_xstart=predict_xstart,rescale_timesteps=rescale_timesteps,
                                                   rescale_learned_sigmas=rescale_learned_sigmas, timestep_respacing='')
        self.schedule_sampler = create_named_schedule_sampler(schedule_sampler, self.diffusion)
        self.reverve_diffusion = create_gaussian_diffusion(steps=diffusion_steps, learn_sigma=learn_sigma, sigma_small=sigma_small, noise_schedule=noise_schedule,
                                                    use_kl=use_kl, predict_xstart=predict_xstart,rescale_timesteps=rescale_timesteps,
                                                    rescale_learned_sigmas=rescale_learned_sigmas, timestep_respacing=timestep_respacing)

        self.item_embedding = item_encoder if share_item_encoder==1 else torch.nn.Embedding(num_items, embed_dim, 0)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.diffusion_loss = 0.0

    def forward(self, batch):
        # encode
        seq_emb = self.item_embedding(batch["in_"+self.fiid])
        non_zero_num = batch["in_"+self.fiid].count_nonzero(dim=1).unsqueeze(-1)
        seq_emb = seq_emb.sum(1) / non_zero_num.pow(0.5)
        z = self.dropout(seq_emb)
        z = seq_emb
        
        #训练第二阶段，训练diffusion
        if self.training:
            t, weights = self.schedule_sampler.sample(z.shape[0], device=z.device)
            terms = self.diffusion.training_losses(self.diffusion_model, z, t, model_kwargs={'x_start':z, 'training':self.training})
            z = terms["z"]
            terms["loss"] = (terms["loss"] * weights).mean() + self.kl_lambda * terms["kl_loss"]
            for k, v in terms.items():
                terms[k] = v.mean()
            self.diffusion_loss = terms
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, terms["loss"].detach()
                )
        else:
            z = self.pc_sampler(z)

        return z

    def pc_sampler(self, x_start):
        sample_fn = (
            self.reverve_diffusion.p_sample_loop if not self.use_ddim else self.reverve_diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            self.diffusion_model,
            (x_start.shape[0], x_start.shape[1]),
            clip_denoised=self.clip_denoised,
            model_kwargs={"x_start": x_start, "training": False},
        )
        return sample


class DDPMVAE(BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('DDPMVAE')
        parent_parser.add_argument("--dropout", type=int, default=0.5, help='dropout rate for MLP layers')
        parent_parser.add_argument("--share_item_encoder", type=int, default=1, help="")

        parent_parser.add_argument("--learn_sigma", type=bool, default=False, help="")
        parent_parser.add_argument("--sigma_small", type=bool, default=False, help="")
        parent_parser.add_argument("--diffusion_steps", type=int, default=1000, help="reverse diffusion steps")
        parent_parser.add_argument("--noise_schedule", type=str, default="linear", help="noise schedule")
        parent_parser.add_argument("--timestep_respacing", type=str, default="", help="timestep respacing")
        parent_parser.add_argument("--use_kl", type=bool, default=False, help="")
        parent_parser.add_argument("--predict_xstart", type=bool, default=False, help="")
        parent_parser.add_argument("--rescale_timesteps", type=bool, default=True, help="")
        parent_parser.add_argument("--rescale_learned_sigmas", type=bool, default=True, help="")
        parent_parser.add_argument("--schedule_sampler", type=str, default="uniform", help="schedule sampler")
        parent_parser.add_argument("--kl_lambda", type=float, default=1e-3, help="")
        parent_parser.add_argument("--clip_denoised", type=bool, default=True, help="")
        parent_parser.add_argument("--use_ddim", type=bool, default=False, help="")
        return parent_parser
            
    def _get_dataset_class():
        return AEDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, 0)#pad for 0

    def _get_query_encoder(self, train_data):
        return DDPMVAEQueryEncoder(train_data.fiid, train_data.num_items, self.embed_dim, self.config['dropout_rate'],
                                        self.item_encoder, self.config['share_item_encoder'], self.config['diffusion_steps'], 
                                        self.config['learn_sigma'], self.config['sigma_small'], self.config['noise_schedule'], self.config['use_kl'], 
                                        self.config['predict_xstart'], self.config['rescale_timesteps'], self.config['rescale_learned_sigmas'],self.config['timestep_respacing'], 
                                        self.config['schedule_sampler'], self.config['kl_lambda'], self.config['clip_denoised'], self.config['use_ddim'])

    def _get_score_func(self):
        return InnerProductScorer()

    def _get_sampler(self, train_data):
        return None

    def _get_loss_func(self):
        self.anneal = 0.0
        return SoftmaxLoss()

    def training_step(self, batch):
        loss = super().training_step(batch)
        all_loss = self.query_encoder.diffusion_loss
        all_loss["rec_loss"] = loss
        all_loss["loss"] = loss + all_loss["loss"]
        return all_loss