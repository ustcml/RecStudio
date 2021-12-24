from typing import Sized, Iterator
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
import numpy as np
import pandas as pd
import os,copy, torch
import scipy.sparse as ssp
from torchrec.utils.utils import parser_yaml
class MFDataset(Dataset):
    def __init__(self, config_path):
        self.config = parser_yaml(config_path)
        self._init_common_field()
        self._load_all_data(self.config['data_dir'], self.config['field_separator'])
        ## first factorize user id and item id, and then filtering to determine the valid user set and item set
        self._filter(self.config['min_user_inter'], self.config['min_item_inter'])
        self._map_all_ids()
        self._post_preprocess()

    def _init_common_field(self):
        self.field2type = {}
        self.field2token2idx = {}
        self.field2tokens = {}
        self.field2maxlen = self.config['field_max_len']  or {}
        self.fuid = self.config['user_id_field'].split(':')[0]
        self.fiid = self.config['item_id_field'].split(':')[0]
        self.ftime = self.config['time_field'].split(':')[0]
        self.frating = self.config['rating_field'].split(':')[0]
        
    def __test__(self):
        feat = self.network_feat[1][-10:]
        #feat = self.inter_feat[-10:]
        print(feat)
        self._map_all_ids()
        feat1 = self._recover_unmapped_feature(self.network_feat[1])
        #feat1 = self._recover_unmapped_feature(self.inter_feat)
        print(feat1[-10:])
        self._prepare_user_item_feat()
        feat2 = self._recover_unmapped_feature(self.network_feat[1])[-10:]
        #feat2 = self._recover_unmapped_feature(self.inter_feat)[-10:]
        #feat2 = feat2[feat2[self.fiid].isin(feat[self.fiid])].sort_values('item_id')
        print(feat2)

    def _filter_ratings(self):
        if self.config['rating_threshold'] is not None:
            if not self.config['drop_low_rating']:
                self.inter_feat[self.frating] = (self.inter_feat[self.frating] > self.config['rating_threshold']).astype(float)
            else:
                self.inter_feat = self.inter_feat[self.inter_feat[self.frating] > self.config['rating_threshold']]
                self.inter_feat[self.frating] = 1.0

    def _load_all_data(self, data_dir, field_sep):
        # load interaction features
        inter_feat_path = os.path.join(data_dir, self.config['inter_feat_name'])
        self.inter_feat = self._load_feat(inter_feat_path, field_sep, self.config['inter_feat_field'])

        # load user features
        self.user_feat = None
        if self.config['user_feat_name'] is not None:
            user_feat = [] 
            for _, user_feat_col in zip(self.config['user_feat_name'], self.config['user_feat_field']):
                user_feat_path = os.path.join(data_dir, _)
                user_f = self._load_feat(user_feat_path, field_sep, user_feat_col)
                user_f.set_index(self.fuid, inplace=True)
                user_feat.append(user_f)
            self.user_feat = pd.concat(user_feat, axis=1)
            self.user_feat.reset_index(inplace=True)
            self._fill_nan(self.user_feat)

        self.item_feat = None
        if self.config['item_feat_name'] is not None:
            # load item features
            item_feat = []
            for _, item_feat_col in zip(self.config['item_feat_name'], self.config['item_feat_field']):
                item_feat_path = os.path.join(data_dir, _)
                item_f = self._load_feat(item_feat_path, field_sep, item_feat_col)
                item_f.set_index(self.fiid, inplace=True)
                item_feat.append(item_f)
            self.item_feat = pd.concat(item_feat, axis=1) # it is possible to generate nan, that should be filled with [pad]
            self.item_feat.reset_index(inplace=True)
            self._fill_nan(self.item_feat)
        
        # load network features
        if self.config['network_feat_name'] is not None:
            self.network_feat = [None] * len(self.config['network_feat_name'])
            self.node_link = [None] * len(self.config['network_feat_name'])
            self.node_relink = [None] * len(self.config['network_feat_name'])
            self.mapped_fields = [_.split(':')[0] for _ in self.config['mapped_feat_field']]
            for i, (name, fields) in enumerate(zip(self.config['network_feat_name'], self.config['network_feat_field'])):
                if len(name) == 2:
                    net_name, link_name = name
                    net_field, link_field = fields
                    link = self._load_feat(os.path.join(data_dir, link_name), \
                        field_sep, link_field, update_dict=False).to_numpy()
                    self.node_link[i] = dict(link)
                    self.node_relink[i] = dict(link[:,[1,0]])
                    feat = self._load_feat(os.path.join(data_dir, net_name), field_sep, net_field)
                    for col in feat.columns[:2]:
                        feat[col] = [self.node_relink[i][id] if id in self.node_relink[i] else id for id in feat[col]] 
                    self.network_feat[i] = feat
                else:
                    net_name, net_field = name[0], fields[0]
                    self.network_feat[i] = self._load_feat(os.path.join(data_dir, net_name), field_sep, net_field)
                
    def _fill_nan(self, feat, mapped=False):
        for field in feat:
            ftype = self.field2type[field]
            if ftype == 'float':
                feat[field].fillna(value=feat[field].mean(), inplace=True)
            elif ftype == 'token':
                feat[field].fillna(value=0 if mapped else '[PAD]', inplace=True)
            else:
                dtype = (np.int64 if mapped else str) if ftype == 'token_seq' else np.float64 
                feat[field] = feat[field].map(lambda x: np.array([], dtype=dtype) if isinstance(x, float) else x)

    def _load_feat(self, feat_path, sep, feat_cols, update_dict=True):
        fields, types_of_fields = zip(*(_.split(':') for _ in feat_cols))
        dtype = (np.float64 if _ == 'float' else str for _ in types_of_fields)
        if update_dict:
            self.field2type.update(dict(zip(fields, types_of_fields)))
        feat = pd.read_csv(feat_path, sep=sep, usecols=fields, dtype=dict(zip(fields, dtype)))[list(fields)]
        seq_sep = self.config['seq_separator']
        for col, t in zip(fields, types_of_fields):
            if not t.endswith('seq'):
                if update_dict and (col not in self.field2maxlen):
                    self.field2maxlen[col] = 1
                continue
            feat[col].fillna(value='', inplace=True)
            cast = float if 'float' in t else str
            feat[col] = feat[col].map(lambda _: np.array(list(map(cast, filter(None, _.split(seq_sep)))), dtype=cast))
            if update_dict and (col not in self.field2maxlen):
                self.field2maxlen[col] = feat[col].map(len).max()
        return feat

    def _get_map_fields(self):
        #fields_share_space = self.config['fields_share_space'] or []
        if self.config['network_feat_name'] is not None:
            network_fields = {col: mf for _, mf in zip(self.network_feat, self.mapped_fields) for col in _.columns[:2]}
        else:
            network_fields = {}
        fields_share_space = [[f] for f, t in self.field2type.items() if ('token' in t) and (f not in network_fields)]
        for k, v in network_fields.items():
            for field_set in fields_share_space:
                if v in field_set:
                    field_set.append(k)
        return fields_share_space

    def _get_feat_list(self):
        ## if we have more features, please add here
        feat_list = [self.inter_feat, self.user_feat, self.item_feat]
        if self.config['network_feat_name'] is not None:
            feat_list.extend(self.network_feat)
        #return list(feat for feat in feat_list if feat is not None)
        return feat_list

    def _map_all_ids(self):
        fields_share_space = self._get_map_fields()
        feat_list = self._get_feat_list()
        for field_set in fields_share_space:
            flag = self.config['network_feat_name'] is not None \
                and (self.fuid in field_set or self.fiid in field_set)
            token_list = []
            field_feat = [(field, feat, idx) for field in field_set \
                for idx, feat in enumerate(feat_list) if (feat is not None) and (field in feat)]
            for field, feat, _ in field_feat:
                if 'seq' not in self.field2type[field]:
                    token_list.append(feat[field].values)
                else:
                    token_list.append(feat[field].agg(np.concatenate))
            count_inter_user_or_item = sum(1 for x in field_feat if x[-1] < 3)
            split_points = np.cumsum([len(_) for _ in token_list])
            token_list = np.concatenate(token_list)
            tid_list, tokens = pd.factorize(token_list)
            max_user_or_item_id = np.max(tid_list[:split_points[count_inter_user_or_item-1]]) + 1 if flag else 0
            if '[PAD]' not in set(tokens):
                tokens = np.insert(tokens, 0, '[PAD]')
                tid_list = np.split(tid_list + 1, split_points[:-1])
                token2id = {tok: i for (i, tok) in enumerate(tokens)}
                max_user_or_item_id += 1
            else:
                token2id = {tok: i for (i, tok) in enumerate(tokens)}
                tid = token2id['[PAD]']
                tokens[tid] = tokens[0]
                token2id[tokens[0]] = tid
                tokens[0] = '[PAD]'
                token2id['[PAD]'] = 0
                idx_0, idx_1 = (tid_list == 0), (tid_list == tid)
                tid_list[idx_0], tid_list[idx_1] = tid, 0
                tid_list = np.split(tid_list, split_points[:-1])


            for (field, feat, idx), _ in zip(field_feat, tid_list):
                if field not in self.field2tokens:
                    if flag:
                        if (field in [self.fuid, self.fiid]):
                            self.field2tokens[field] = tokens[:max_user_or_item_id]
                            self.field2token2idx[field] = {tokens[i]: i for i in range(max_user_or_item_id)}
                        else:
                            tokens_ori = self._get_ori_token(idx-3, tokens)
                            self.field2tokens[field] = tokens_ori
                            self.field2token2idx[field] = {t:i for i,t in enumerate(tokens_ori)}
                    else:
                        self.field2tokens[field] = tokens
                        self.field2token2idx[field] = token2id
                if 'seq' not in self.field2type[field]:
                    feat[field] = _
                    feat[field] = feat[field].astype('Int64')
                else:
                    sp_point = np.cumsum(feat[field].agg(len))[:-1]
                    feat[field] = np.split(_, sp_point)
    
    def _get_ori_token(self, idx, tokens):
        if self.node_link[idx] is not None:
            return [self.node_link[idx][tok] if tok in self.node_link[idx] else tok for tok in tokens]
        else:
            return tokens

    def _prepare_user_item_feat(self):
        if self.user_feat is not None:
            self.user_feat.set_index(self.fuid, inplace=True)
            self.user_feat = self.user_feat.reindex(np.arange(self.num_users))
            self.user_feat.reset_index(inplace=True)
            self._fill_nan(self.user_feat, mapped=True)
        else:
            self.user_feat = pd.DataFrame({self.fuid: np.arange(self.num_users)})

        if self.item_feat is not None:
            self.item_feat.set_index(self.fiid, inplace=True)
            self.item_feat = self.item_feat.reindex(np.arange(self.num_items))
            self.item_feat.reset_index(inplace=True)
            self._fill_nan(self.item_feat, mapped=True)
        else:
            self.item_feat = pd.DataFrame({self.fiid: np.arange(self.num_items)})
    
    def _post_preprocess(self):
        if self.ftime in self.inter_feat:
            if self.field2type[self.ftime] == 'float':
                self.inter_feat.sort_values(by=[self.fuid, self.ftime], inplace=True)
                self.inter_feat.reset_index(drop=True, inplace=True)
            else:
                raise ValueError('The field [{self.ftime}] should be float type')
        self._prepare_user_item_feat()

    def _recover_unmapped_feature(self, feat):
        feat = feat.copy()
        for field in feat:
            if field in self.field2tokens:
                feat[field] = feat[field].map(lambda x: self.field2tokens[field][x])
        return feat

    def _filter(self, min_user_inter, min_item_inter):
        self._filter_ratings()
        item_list = self.inter_feat[self.fiid]
        item_idx_list, items = pd.factorize(item_list)
        user_list = self.inter_feat[self.fuid]
        user_idx_list, users = pd.factorize(user_list)
        user_item_mat = ssp.csc_matrix((np.ones_like(user_idx_list), (user_idx_list, item_idx_list)))
        cols = np.arange(items.size)
        rows = np.arange(users.size)
        while(True):
            m, n = user_item_mat.shape
            col_sum = np.squeeze(user_item_mat.sum(axis=0).A)
            col_ind = col_sum >= min_item_inter
            col_count = np.count_nonzero(col_ind)
            if col_count > 0:
                cols = cols[col_ind]
                user_item_mat = user_item_mat[:,col_ind]
            row_sum = np.squeeze(user_item_mat.sum(axis=1).A)
            row_ind = row_sum >= min_user_inter
            row_count = np.count_nonzero(row_ind)
            if row_count > 0:
                rows = rows[row_ind]
                user_item_mat = user_item_mat[row_ind, :]
            if col_count == n and row_count == m:
                break
            else:
                pass
                #@todo add output info if necessary
        
        keep_users = set(users[rows])
        keep_items = set(items[cols])
        keep = user_list.isin(keep_users)
        keep &= item_list.isin(keep_items)
        self.inter_feat = self.inter_feat[keep]
        self.inter_feat.reset_index(drop=True, inplace=True)
        #if self.user_feat is not None:
        #    self.user_feat = self.user_feat[self.user_feat[self.fuid].isin(keep_users)]
        #    self.user_feat.reset_index(drop=True, inplace=True)
        #if self.item_feat is not None:
        #    self.item_feat = self.item_feat[self.item_feat[self.fiid].isin(keep_items)]
        #    self.item_feat.reset_index(drop=True, inplace=True)
 
    def get_graph(self, idx, value_field=None):
        from scipy.sparse import csc_matrix
        if idx == 0:
            source_field = self.fuid
            target_field = self.fiid
            feat = self.inter_feat[self.inter_feat_subset]
        else:
            if self.network_feat is not None:
                if idx - 1 < len(self.network_feat):
                    feat = self.network_feat[idx - 1]
                    source_field, target_field = feat.columns[:2]
                else:
                    raise ValueError(f'idx [{idx}] is larger than the number of network features [{len(self.network_feat)}] minus 1' )
            else:
                raise ValueError(f'No network feature is input while idx [{idx}] is larger than 1')
        
        rows = feat[source_field]
        cols = feat[target_field]
        if value_field is not None:
            if value_field in feat:
                vals = feat[value_field]
            else:
                raise ValueError(f'valued_field [{value_field}] does not exist')
        else:
            vals = np.ones(len(rows))
        return csc_matrix((vals, (rows, cols)), (self.num_values(source_field), self.num_values(target_field)))

    def _split_by_ratio(self, ratio, data_count, user_mode):
        m = len(data_count)
        if not user_mode:
            splits = np.outer(data_count, ratio).astype(np.int32)
            splits[:,0] = data_count - splits[:,1:].sum(axis=1)
            for i in range(1, len(ratio)):
                idx = (splits[:, -i] == 0) & (splits[:, 0] > 1)
                splits[idx, -i] += 1
                splits[idx, 0] -= 1
        else:
            idx = np.random.permutation(m)
            sp_ = (m * np.array(ratio)).astype(np.int32)
            sp_[0] = m - sp_[1:].sum()
            sp_ = sp_.cumsum()
            parts = np.split(idx, sp_[:-1])
            splits = np.zeros((m, len(ratio)), dtype=np.int32)
            for _, p in zip(range(len(ratio)), parts):
                splits[p, _] = data_count.iloc[p]

        splits = np.hstack([np.zeros((m, 1), dtype=np.int32), np.cumsum(splits, axis=1)])
        cumsum = np.hstack([[0], data_count.cumsum()[:-1]])
        splits = cumsum.reshape(-1, 1) + splits
        return splits, data_count.index if m > 1 else None
    
    def _split_by_leave_one_out(self, leave_one_num, data_count, rep=True):
        m = len(data_count)
        cumsum = data_count.cumsum()[:-1]
        if rep:
            splits = np.ones((m, leave_one_num + 1), dtype=np.int32)
            splits[:,0] = data_count - leave_one_num
            for _ in range(leave_one_num):
                idx = splits[:, 0] < 1
                splits[idx, 0] += 1
                splits[idx, _] -= 1
            splits = np.hstack([np.zeros((m, 1), dtype=np.int32), np.cumsum(splits, axis=1)])
        else:
            def get_splits(bool_index):
                idx = bool_index.values.nonzero()[0]
                if len(idx) > 2:
                    return [0, idx[-2], idx[-1], len(idx)]
                elif len(idx) == 2:
                    return [0, idx[-1], idx[-1], len(idx)]
                else:
                    return [0, len(idx), len(idx), len(idx)]
            splits = np.array([get_splits(bool_index) for bool_index in np.split(self.first_item_idx, cumsum)])
        
        cumsum = np.hstack([[0], cumsum])
        splits = cumsum.reshape(-1, 1) + splits
        return splits, data_count.index if m > 1 else None
                
    
    def _get_data_idx(self, splits):
        splits, uids = splits
        data_idx = [list(zip(splits[:, i-1], splits[:, i])) for i in range(1, splits.shape[1])]
        if not getattr(self, 'fmeval', False):
            if uids is not None:
                d = [torch.from_numpy(np.hstack([np.arange(*e) for e in data_idx[0]]))]
                for _ in data_idx[1:]:
                    d.append(torch.tensor([[u, *e] for u, e in zip(uids, _)]))
                return d
            else:
                d = [torch.from_numpy(np.hstack([np.arange(*e) for e in data_idx[0]]))]
                for _ in data_idx[1:]:
                    start, end = _[0]
                    data = self.inter_feat.get_col(self.fuid)[start:end]
                    uids, counts = data.unique_consecutive(return_counts=True)
                    cumsum = torch.hstack([torch.tensor([0]), counts.cumsum(-1)]) + start
                    d.append(torch.tensor([[u, st, en] for u, st, en in zip(uids, cumsum[:-1], cumsum[1:])]))
                #data_idx = [torch.from_numpy(np.hstack([np.arange(*e) for e in _])) for _ in data_idx]
                return d
        else:
            return [torch.from_numpy(np.hstack([np.arange(*e) for e in _])) for _ in data_idx]
    
    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, index):
        if self.data_index.dim() > 1:
            idx = self.data_index[index]
            data = {self.fuid: idx[:,0]}
            data.update(self.user_feat[data[self.fuid]])
            start = idx[:, 1]
            end = idx[:, 2]
            lens = end - start
            l = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
            d = self.inter_feat.get_col(self.fiid)[l]
            rating = self.inter_feat.get_col(self.frating)[l]
            data[self.fiid] = pad_sequence(d.split(tuple(lens.numpy())), batch_first=True)
            data[self.frating] = pad_sequence(rating.split(tuple(lens.numpy())), batch_first=True)
        else:
            idx = self.data_index[index]
            data = self.inter_feat[idx]
            uid, iid = data[self.fuid], data[self.fiid]
            data.update(self.user_feat[uid])
            data.update(self.item_feat[iid])
        
        if getattr(self, 'eval_mode', False) and 'user_hist' not in data:
            user_count = self.user_count[data[self.fuid]].max()
            data['user_hist'] = self.user_hist[data[self.fuid]][:, 0:user_count]

        return data

    def _copy(self, idx):
        d = copy.copy(self)
        d.data_index = idx
        return d

    
    """ split interactions in entrywise, used for mf-like model
    Args:
        ratio (np.Array): ratio of spliting dataset
        order (bool, optional): 
        split_mode: user_entry, entry, user
    """
    def build(self, ratio_or_num, shuffle=True, split_mode='user_entry', fmeval=False):
        self.fmeval = fmeval
        return self._build(ratio_or_num, shuffle, split_mode, True, False)

    def _build(self, ratio_or_num, shuffle, split_mode, drop_dup, rep):
        ## for general recommendation, only support non-repetive recommendation
        ## keep first data, sorted by time or not, split by user or not
        if not hasattr(self, 'first_item_idx'):
            self.first_item_idx = ~self.inter_feat.duplicated(subset=[self.fuid, self.fiid], keep='first')
        if drop_dup:
            self.inter_feat = self.inter_feat[self.first_item_idx]
        
        if split_mode == 'user_entry':
            user_count = self.inter_feat[self.fuid].groupby(self.inter_feat[self.fuid], sort=False).count()
            if shuffle:
                cumsum = np.hstack([[0], user_count.cumsum()[:-1]])
                idx = np.concatenate([np.random.permutation(c) + start for start, c in zip(cumsum, user_count)])
                self.inter_feat = self.inter_feat.iloc[idx].reset_index(drop=True)
        elif split_mode == 'entry':
            if shuffle:
                self.inter_feat = self.inter_feat.sample(frac=1).reset_index(drop=True)
            user_count = np.array([len(self.inter_feat)])
        elif split_mode == 'user':
            user_count = self.inter_feat[self.fuid].groupby(self.inter_feat[self.fuid], sort=False).count()
        
        if isinstance(ratio_or_num, int):
            splits = self._split_by_leave_one_out(ratio_or_num, user_count, rep)
        else:
            splits = self._split_by_ratio(ratio_or_num, user_count, split_mode == 'user')
        
        if split_mode == 'entry':
            splits_ = splits[0][0]
            for start, end in zip(splits_[:-1], splits_[1:]):
                self.inter_feat[start:end] = self.inter_feat[start:end].sort_values(by=self.fuid)

        
        #if isinstance(self, AEDataset) or isinstance(self, SeqDataset):
        #    self.inter_feat.drop(self.fuid, inplace=True, axis=1)
        self.dataframe2tensors()
        datasets = [self._copy(_) for _ in self._get_data_idx(splits)]
        user_hist, user_count = datasets[0].get_hist(True)
        for d in datasets[:2]:
            d.user_hist = user_hist
            d.user_count = user_count
        if len(datasets) > 2:
            assert len(datasets) == 3 
            uh, uc = datasets[1].get_hist(True)
            uh = torch.cat((user_hist, uh), dim=-1).sort(dim=-1, descending=True).values
            uc = uc + user_count
            datasets[-1].user_hist = uh
            datasets[-1].user_count = uc
        return datasets

    def dataframe2tensors(self):
        self.inter_feat = TensorFrame.fromPandasDF(self.inter_feat, self)
        self.user_feat = TensorFrame.fromPandasDF(self.user_feat, self)
        self.item_feat = TensorFrame.fromPandasDF(self.item_feat, self)
        if hasattr(self, 'network_feat'):
            for i in range(len(self.network_feat)):
                self.network_feat[i] = TensorFrame.fromPandasDF(self.network_feat[i], self)
    
    def train_loader(self, batch_size, shuffle=True, num_workers=1, drop_last=False, load_combine=False):
        if not hasattr(self, 'loaders'):
            return self.loader(batch_size, shuffle, num_workers, drop_last)
        else:
            loaders = [l(batch_size, shuffle, num_workers, drop_last) if callable(l) else l for l in self.loaders]
            if load_combine:
                return loaders
            else:
                return ChainedDataLoader(loaders, getattr(self, 'nepoch', None))

    def loader(self, batch_size, shuffle=True, num_workers=1, drop_last=False):
        if self.data_index.dim() > 1: # has sample_length
            sampler = SortedDataSampler(self, batch_size, shuffle, drop_last)
        else:
            sampler = DataSampler(self, batch_size, shuffle, drop_last)
        output = DataLoader(self, sampler=sampler, batch_size=None, shuffle=False, num_workers=num_workers)
        return output
    
    @property
    def sample_length(self):
        if self.data_index.dim() > 1:
            return self.data_index[:, 2] - self.data_index[:, 1]
        else:
            raise ValueError('can not compute sample length for this dataset')

    def eval_loader(self, batch_size, num_workers=1):
        if not getattr(self, 'fmeval', False):
            self.eval_mode = True
            sampler = SortedDataSampler(self, batch_size)
            output = DataLoader(self, sampler=sampler, batch_size=None, shuffle=False, num_workers=num_workers)
            return output
        else:
            return self.loader(batch_size, shuffle=False, num_workers=num_workers)
    
    def drop_feat(self, keep_fields):
        if keep_fields is not None and len(keep_fields) > 0:
            fields = set(keep_fields)
            fields.add(self.frating)
            for feat in self._get_feat_list():
                feat.del_fields(fields)
            if 'user_hist' in fields:
                self.user_feat.add_field('user_hist', self.user_hist)
            if 'item_hist' in fields:
                self.item_feat.add_field('item_hist', self.get_hist(False))
    
    def get_hist(self, isUser=True):
        user_array = self.inter_feat.get_col(self.fuid)[self.inter_feat_subset]
        item_array = self.inter_feat.get_col(self.fiid)[self.inter_feat_subset]
        sorted, index = torch.sort(user_array if isUser else item_array)
        user_item, count = torch.unique_consecutive(sorted, return_counts=True)
        list_ = torch.split(item_array[index] if isUser else user_array[index], tuple(count.numpy()))
        tensors = [torch.tensor([], dtype=torch.int64) for _ in range(self.num_users if isUser else self.num_items)]
        for i, l in zip(user_item, list_):
            tensors[i] = l
        #tensors = np.array(tensors, dtype=object)
        user_count = torch.tensor([len(e) for e in tensors])
        tensors = pad_sequence(tensors, batch_first=True)
        return tensors, user_count


    @property
    def inter_feat_subset(self):
        if self.data_index.dim() > 1:
            return torch.cat([torch.arange(s, e) for s, e in zip(self.data_index[:,1], self.data_index[:, 2])])
        else:
            return self.data_index

    @property
    def item_freq(self):
        if not hasattr(self, 'data_index'):
            raise ValueError('please build the dataset first by call the build method')
        l = self.inter_feat.get_col(self.fiid)[self.inter_feat_subset]
        it, count = torch.unique(l, return_counts=True)
        it_freq = torch.zeros(self.num_items, dtype=torch.int64)
        it_freq[it] = count
        return it_freq

    @property
    def num_users(self):
        return self.num_values(self.fuid)
    
    @property
    def num_items(self):
        return self.num_values(self.fiid)
    
    @property
    def num_inters(self):
        return len(self.inter_feat)

    def num_values(self, field):
        if 'token' not in self.field2type[field]:
            return self.field2maxlen[field]
        else:
            return len(self.field2tokens[field])
        
class AEDataset(MFDataset):
    def build(self, ratio_or_num, shuffle=False):
        return self._build(ratio_or_num, shuffle, 'user_entry', True, False)
    
    def _get_data_idx(self, splits):
        splits, uids = splits
        data_idx = [list(zip(splits[:, i-1], splits[:, i])) for i in range(1, splits.shape[1])]
        #data_idx = [np.array([(u, slice(*e)) for e, u in zip(_, uids)]) for _ in data_idx]
        data_idx = [torch.tensor([[u, *e] for e, u in zip(_, uids)]) for _ in data_idx]
        #data = [np.array(list(zip(data_idx[0], data_idx[i]))) for i in range(len(data_idx))]
        data = [torch.cat((data_idx[0], data_idx[i]), -1) for i in range(len(data_idx))]
        return data

    def __getitem__(self, index):
        idx = self.data_index[index]
        #data = {self.fuid: idx[0][0]}
        data = {self.fuid: idx[:,0]}
        data.update(self.user_feat[data[self.fuid]])
        for i, n in enumerate(['in_', '']):
            start = idx[:, i * 3 + 1]
            end = idx[:, i * 3 + 2]
            lens = end - start
            l = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
            d = self.inter_feat[l]
            for k in d:
                d[k] = pad_sequence(d[k].split(tuple(lens.numpy())), batch_first=True)
            d.update(self.item_feat[d[self.fiid]])
            #d = self.get_value(self.inter_feat, idx[-1])
            #d.update(self.get_value(self.item_feat, d[self.fiid]))
            for k, v in d.items():
                if k != self.fuid:
                    data[n+k] = v
        
        if getattr(self, 'eval_mode', False) and 'user_hist' not in data:
            data['user_hist'] = data['in_item_id']
        
        return data

    @property
    def inter_feat_subset(self):
        index = torch.cat([torch.arange(s, e) for s, e in zip(self.data_index[:,-2], self.data_index[:, -1])])
        return index

class SeqDataset(MFDataset):
    def build(self, ratio_or_num, rep=True, train_rep=True):
        self.test_rep = rep
        self.train_rep = train_rep if not rep else True
        return self._build(ratio_or_num, False, 'user_entry', False, rep)

    def _get_data_idx(self, splits):
        splits, uids = splits
        maxlen = self.config['max_seq_len'] or (splits[:,-1] - splits[:,0]).max()
        def keep_first_item(dix, part):
            if ((dix == 0) and self.train_rep) or ((dix > 0) and self.test_rep):
                return part
            else:
                return part[self.first_item_idx.iloc[part[:,-1]].values]
        def get_slice(sp, u):
            #data = [(u, slice(max(sp[0], i - maxlen), i), i) for i in range(sp[0], sp[-1])]
            data = np.array([[u, max(sp[0], i - maxlen), i] for i in range(sp[0], sp[-1])], dtype=np.int64)
            sp -= sp[0]
            return np.split(data[1:], sp[1:-1]-1)
        output = [get_slice(sp, u) for sp, u in zip(splits, uids)]
        output = [torch.from_numpy(np.concatenate(_)) for _ in zip(*output)]
        output = [keep_first_item(dix, _) for dix, _ in enumerate(output)]
        return output

    def __getitem__(self, index):
        #uid, source, target = self.data_index[index]
        idx = self.data_index[index]
        data = {self.fuid: idx[:, 0]}
        data.update(self.user_feat[data[self.fuid]])
        target_data = self.inter_feat[idx[:, 2]]
        target_data.update(self.item_feat[target_data[self.fiid]])
        #source_data = self.get_value(self.inter_feat, idx[:, 1])
        start = idx[:, 1]
        end = idx[:, 2]
        lens = end - start
        data['seqlen'] = lens
        l = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
        source_data = self.inter_feat[l]
        for k in source_data:
            source_data[k] = pad_sequence(source_data[k].split(tuple(lens.numpy())), batch_first=True)
        source_data.update(self.item_feat[source_data[self.fiid]])

        for n, d in zip(['in_', ''], [source_data, target_data]):
            for k, v in d.items():
                if k != self.fuid:
                    data[n+k] = v
        
        if getattr(self, 'eval_mode', False) and 'user_hist' not in data:
            user_count = self.user_count[data[self.fuid]].max()
            data['user_hist'] = self.user_hist[data[self.fuid]][:, 0:user_count]

        return data
    
    @property
    def inter_feat_subset(self):
        return self.data_index[:, -1]
        

               
class TensorFrame(Dataset):
    @classmethod
    def fromPandasDF(cls, dataframe, dataset):
        data = {}
        length = len(dataframe.index)
        for field in dataframe:
            ftype = dataset.field2type[field]
            value = dataframe[field]
            if ftype == 'token_seq':
                    seq_data = [torch.from_numpy(d[:dataset.field2maxlen[field]]) for d in value]
                    data[field] = pad_sequence(seq_data, batch_first=True)
            elif ftype == 'float_seq':
                seq_data = [torch.from_numpy(d[:dataset.field2maxlen[field]]) for d in value]
                data[field] = pad_sequence(seq_data, batch_first=True)
            elif ftype == 'token':
                data[field] = torch.from_numpy(dataframe[field].to_numpy(np.int64))
            else:
                data[field] = torch.from_numpy(dataframe[field].to_numpy(np.float32))
        return cls(data, length)

    def __init__(self, data, length):
        self.data = data
        self.length = length

    def get_col(self, field):
        return self.data[field]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        ret = {}
        for field, value in self.data.items():
            ret[field] = value[idx]
        return ret
    
    def del_fields(self, keep_fields):
        for f in self.fields:
            if f not in keep_fields:
                del self.data[f]
    
    def loader(self, batch_size, shuffle=False, num_workers=1, drop_last=False):
        sampler = DataSampler(self, batch_size, shuffle, drop_last)
        output = DataLoader(self, sampler=sampler, batch_size=None, shuffle=False, num_workers=num_workers)
        return output

    def add_field(self, field, value):
        self.data[field] = value

    def reindex(self, idx):
        output = copy.deepcopy(self)
        for f in output.fields:
            output.data[f] = output.data[f][idx]
        return output

    @property
    def fields(self):
        return set(self.data.keys())


class DataSampler(Sampler):
    def __init__(self, data_source:Sized, batch_size, shuffle=True, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator
    
    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        if self.shuffle:
            output = torch.randperm(n, generator=generator).split(self.batch_size)
        else:
            output = torch.arange(n).split(self.batch_size)
        if self.drop_last and len(output[-1]) < self.batch_size:
            yield from output[:-1]
        else:
            yield from output

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class SortedDataSampler(Sampler):
    def __init__(self, data_source:Sized, batch_size, shuffle=False, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator
    
    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            output = torch.randperm(n) // (self.batch_size * 10)
            output = self.data_source.sample_length + output * (self.data_source.sample_length.max() + 1)
        else:
            output = self.data_source.sample_length
        output = torch.sort(output).indices
        output = output.split(self.batch_size)
        if self.drop_last and len(output[-1]) < self.batch_size:
            yield from output[:-1]
        else:
            yield from output

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size
    

class ChainedDataLoader:
    def __init__(self, loaders, nepoch=None) -> None:
        self.loaders = loaders
        self.epoch = -1
        nepoch = np.ones(len(loaders)) if nepoch is None else np.array(nepoch)
        self.iter_idx = np.concatenate([np.repeat(i, c) for i, c in enumerate(nepoch)])
    
    def __iter__(self):
        self.epoch += 1
        return iter(self.loaders[self.iter_idx[self.epoch % len(self.iter_idx)]])

