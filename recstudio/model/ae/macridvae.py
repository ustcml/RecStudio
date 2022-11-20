import torch
from recstudio.data.dataset import AEDataset
from recstudio.model.basemodel import BaseRetriever, Recommender
from recstudio.model.loss_func import SoftmaxLoss
from recstudio.model.module import MLPModule
from recstudio.model.scorer import MacridVAEScorer
import torch.nn.functional as F

class MacridVAEDecoder(torch.nn.Module):
    def __init__(self, num_items, embed_dim) -> None:
        super().__init__()
        self.item_vector =torch.nn.Embedding(num_items, embed_dim, 0)#0 for padding
        self.items_idx = None
    
    def forward(self, batch):
        self.items_idx = batch
        return self



class MacridVAEQueryEncoder(torch.nn.Module):
    def __init__(self, fiid, num_items, embed_dim, dropout_rate,
                 encoder_dims, activation, kfac, tau, nogb, std, item_encoder, reg_weights):
        super().__init__()

        self.fiid = fiid
        self.kfac = kfac
        self.tau = tau
        self.nogb = nogb
        self.std = std
        self.regs = reg_weights
        self.item_encoder = item_encoder
        self.k_embedding = torch.nn.Embedding(kfac, embed_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, 0)
        self.encoders = torch.nn.Sequential(
            MLPModule([embed_dim]+encoder_dims[:-1], activation, dropout_rate),
            torch.nn.Linear(([embed_dim]+encoder_dims[:-1])[-1], encoder_dims[-1]*2)
        )
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros_like(std).normal_(mean=0, std=self.std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.

        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        loss = 0
        for name, parm in self.named_parameters():
            if name.endswith('weight'):
                loss = loss + self.regs * parm.norm(2)
        return loss

    def forward(self, batch):
        #对ci进行预测
        cores = F.normalize(self.k_embedding.weight, dim=1)#(k,embed_dim)
        items = F.normalize(self.item_encoder.weight, dim=1)#(num_items, embed_dim)
        cates_logits = torch.matmul(items, cores.transpose(0,1)) / self.tau#(num_items, k), prob for each item belonging to which category

        if self.nogb:
            self.cates = torch.softmax(cates_logits, dim=-1).T
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            self.cates = (self.training * cates_sample + (1-self.training) * cates_mode).T#(k,num_items), at test time, c_i is set to the mode

        zk_list=[]
        mulist = []
        logvarlist = []
        for k in range(self.kfac):#为user生成每个z_k
            cates_k = self.cates[k,:].reshape(1,-1)#(1, num_items),The probability that each item belongs to category k，c_i,k
            cates_batch = cates_k.expand(batch["in_"+self.fiid].shape[0],cates_k.shape[1])#(batch, num_items)
            cates_pos = cates_batch.gather(1,batch["in_"+self.fiid])#(batch, pos_items)
            
            seq_emb = self.item_embedding(batch["in_"+self.fiid])#(batch, pos_items, dim)

            cates_pos[batch["in_"+self.fiid] == 0] = 0.0
            regulizer = cates_pos.pow(2).sum(1).reshape(-1,1)
            seq_emb = (seq_emb * cates_pos.unsqueeze(dim=-1)).sum(1) / regulizer.pow(0.5)

            encoder_h = self.encoders(seq_emb)
            mu, logvar = encoder_h.tensor_split(2, dim=-1)

            mu = F.normalize(mu, dim=1)
            mulist.append(mu)
            logvarlist.append(logvar)

            z_k = self.reparameterize(mu,logvar)#reparameterize，(mu, (sigma * sigma_0)**2)
            zk_list.append(z_k)

        self.z = torch.stack(zk_list, dim=1)
        
        self.kl_loss = None
        # Trick: KL is constant w.r.t. to mu after we normalize mu.
        for i in range(self.kfac):
            kl_ = -0.5 * torch.mean(torch.sum(1 + logvarlist[i] - logvarlist[i].exp(), dim=1))
            self.kl_loss = (kl_ if (self.kl_loss is None) else (self.kl_loss + kl_))

        self.regloss = None
        if self.regs>0.0:
            self.regloss = self.reg_loss()

        return self #(batch, k, dim), #(k,num_items)



class MacridVAE(BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('MacridVAE')
        parent_parser.add_argument("--dropout_rate", type=int, default=0.5, help='dropout rate for MLP layers')
        parent_parser.add_argument("--encoder_dims", type=int, nargs='+', default=64, help='MLP layer size for encoder')
        parent_parser.add_argument("--activation", type=str, default='tanh', help='activation function for MLP layers')
        parent_parser.add_argument("--anneal_max", type=float, default=1.0, help="max anneal coef for KL loss")
        parent_parser.add_argument("--anneal_total_step", type=int, default=2000, help="total anneal steps")
        parent_parser.add_argument("--kfac", type=int, default=7, help='Number of facets (macro concepts).')
        parent_parser.add_argument("--tau", type=float, default=0.1, help='Temperature of sigmoid/softmax, in (0,oo).')
        parent_parser.add_argument("--std", type=float, default=0.075, help='Standard deviation of the Gaussian prior.')
        parent_parser.add_argument("--nogb", type=bool, default=False, help='Disable Gumbel-Softmax sampling.')
        parent_parser.add_argument("--reg_weights", type=float, default=0.0, help='L2 regularization.')
        return parent_parser

    def _init_model(self, train_data):
        super()._init_model(train_data)

    def _get_dataset_class():
        return AEDataset

    def _get_item_encoder(self, train_data):
        return MacridVAEDecoder(train_data.num_items, self.embed_dim)

    def _get_item_vector(self):
        #self.item_encoder.items_idx = None
        #self.item_encoder.item_vec = self.item_encoder.item_vector.weight[1:]
        return self.item_encoder.item_vector.weight[1:]

    def _get_query_encoder(self, train_data):
        return MacridVAEQueryEncoder(train_data.fiid, train_data.num_items,
                                    self.embed_dim, self.config['dropout_rate'], self.config['encoder_dims'],self.config['activation'],self.config['kfac'],
                                    self.config['tau'],self.config['nogb'],self.config['std'], self.item_encoder.item_vector,self.config["reg_weights"])

    def _get_score_func(self):
        return MacridVAEScorer()

    def _get_sampler(self, train_data):
        return None


    def _get_loss_func(self):
        self.anneal = 0.0
        return SoftmaxLoss()

    def training_step(self, batch):
        loss = super().training_step(batch)
        # anneal = min(self.config['anneal_max'], self.anneal)
        # self.anneal = min(self.config['anneal_max'],
        #                   self.anneal + (1.0 / self.config['anneal_total_step']))
        if self.query_encoder.regloss:
            return loss + self.config['anneal_max'] * self.query_encoder.kl_loss + self.query_encoder.regloss
        else:
            return loss + self.config['anneal_max'] * self.query_encoder.kl_loss
