import copy
import os
import pickle
import logging
from operator import itemgetter
from typing import List, Sized, Dict, Optional, Iterator, Union

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
from recstudio.ann.sampler import uniform_sampling
from recstudio.utils import (DEFAULT_CACHE_DIR, check_valid_dataset, set_color,
                             md5, parser_yaml, get_dataset_default_config)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


class MFDataset(Dataset):
    r""" Dataset for Matrix Factorized Methods.

    The basic dataset class in RecStudio.
    """

    def __init__(self, name: str = 'ml-100k', config: Union[Dict, str] = None):
        r"""Load all data.

        Args:
            config(str): config file path or config dict for the dataset.

        Returns:
            recstudio.data.dataset.MFDataset: The ingredients list.
        """
        self.name = name

        self.logger = logging.getLogger('recstudio')

        self.config = get_dataset_default_config(name)
        if config is not None:
            if isinstance(config, str):
                self.config.update(parser_yaml(config))
            elif isinstance(config, Dict):
                self.config.update(config)
            else:
                raise TypeError("expecting `config` to be Dict or string,"
                                f"while get {type(config)} instead.")

        cache_flag, data_dir = check_valid_dataset(self.name, self.config)
        if cache_flag:
            self.logger.info("Load dataset from cache.")
            self._load_cache(data_dir)
        else:
            self._init_common_field()
            self._load_all_data(data_dir, self.config['field_separator'])
            # first factorize user id and item id, and then filtering to
            # determine the valid user set and item set
            self._filter(self.config['min_user_inter'],
                         self.config['min_item_inter'])
            self._map_all_ids()
            self._post_preprocess()
            if self.config['save_cache']:
                self._save_cache(md5(self.config))

        self._use_field = set([self.fuid, self.fiid, self.frating])

    @property
    def field(self):
        return set(self.field2type.keys())

    @property
    def use_field(self):
        return self._use_field

    @use_field.setter
    def use_field(self, fields):
        self._use_field = set(fields)

    @property
    def drop_dup(self):
        return True

    def _load_cache(self, path):
        with open(path, 'rb') as f:
            download_obj = pickle.load(f)
        for k in download_obj.__dict__:
            attr = getattr(download_obj, k)
            setattr(self, k, attr)

    def _save_cache(self, md: str):
        cache_dir = os.path.join(DEFAULT_CACHE_DIR, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, md), 'wb') as f:
            pickle.dump(self, f)

    def _init_common_field(self):
        r"""Inits several attributes.
        """
        self.field2type = {}
        self.field2token2idx = {}
        self.field2tokens = {}
        self.field2maxlen = self.config['field_max_len'] or {}
        self.fuid = self.config['user_id_field'].split(':')[0]
        self.fiid = self.config['item_id_field'].split(':')[0]
        self.ftime = self.config['time_field'].split(':')[0]
        if self.config['rating_field'] is not None:
            self.frating = self.config['rating_field'].split(':')[0]
        else:
            self.frating = None

    def __test__(self):
        feat = self.network_feat[1][-10:]
        print(feat)
        self._map_all_ids()
        feat1 = self._recover_unmapped_feature(self.network_feat[1])
        print(feat1[-10:])
        self._prepare_user_item_feat()
        feat2 = self._recover_unmapped_feature(self.network_feat[1])[-10:]
        print(feat2)

    def __repr__(self):
        info = {"item": {}, "user": {}, "interaction": {}}
        feat = {"item": self.item_feat, "user": self.user_feat, "interaction": self.inter_feat}
        max_num_fields = 0
        max_len_field = max([len(f) for f in self.field]+[len("token_seq")]) + 1

        for k in info:
            info[k]['field'] = list(feat[k].fields)
            info[k]['type'] = [self.field2type[f] for f in info[k]['field']]
            info[k]['##'] = [str(self.num_values(f)) if "token" in t else "-"
                             for f, t in zip(info[k]['field'], info[k]['type'])]
            max_num_fields = max(max_num_fields, len(info[k]['field'])) + 1

        info_str = f"\n{set_color('Dataset Info','green')}: \n"
        info_str += "\n" + "=" * (max_len_field*max_num_fields) + '\n'
        for k in info:
            info_str += set_color(k + ' information: \n', 'blue')
            for k, v in info[k].items():
                info_str += "{}".format(set_color(k, 'yellow')) + " " * (max_len_field-len(k))
                info_str += "".join(["{}".format(i)+" "*(max_len_field-len(i)) for i in v])
                info_str += "\n"
            info_str += "=" * (max_len_field*max_num_fields) + '\n'
        info_str += "{}: {}\n".format(set_color('Total Interactions', 'blue'), self.num_inters)
        info_str += "{}: {:.6f}\n".format(set_color('Sparsity', 'blue'),
                                          (1-self.num_inters / ((self.num_items-1)*(self.num_users-1))))
        info_str += "=" * (max_len_field*max_num_fields)
        return info_str

    def _filter_ratings(self):
        r"""Filter out the interactions whose rating is below `rating_threshold` in config."""
        if self.config['rating_threshold'] is not None:
            if not self.config['drop_low_rating']:
                self.inter_feat[self.frating] = (
                    self.inter_feat[self.frating] >= self.config['rating_threshold']).astype(float)
            else:
                self.inter_feat = self.inter_feat[self.inter_feat[self.frating]
                                                  >= self.config['rating_threshold']]
                self.inter_feat[self.frating] = 1.0

    def _load_all_data(self, data_dir, field_sep):
        r"""Load features for user, item, interaction and network."""
        # load interaction features
        inter_feat_path = os.path.join(
            data_dir, self.config['inter_feat_name'])
        self.inter_feat = self._load_feat(
            inter_feat_path, self.config['inter_feat_header'], field_sep, self.config['inter_feat_field'])
        self.inter_feat = self.inter_feat.dropna(how="any")
        if self.frating is None:
            # add ratings when implicit feedback
            self.frating = 'rating'
            self.inter_feat.insert(0, self.frating, 1)
            self.field2type[self.frating] = 'float'
            self.field2maxlen[self.frating] = 1

        # load user features
        self.user_feat = None
        if self.config['user_feat_name'] is not None:
            user_feat = []
            for _, user_feat_col in zip(self.config['user_feat_name'], self.config['user_feat_field']):
                user_feat_path = os.path.join(data_dir, _)
                user_f = self._load_feat(
                    user_feat_path, self.config['user_feat_header'], field_sep, user_feat_col)
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
                item_f = self._load_feat(
                    item_feat_path, self.config['item_feat_header'], field_sep, item_feat_col)
                item_f.set_index(self.fiid, inplace=True)
                item_feat.append(item_f)
            # it is possible to generate nan, that should be filled with [pad]
            self.item_feat = pd.concat(item_feat, axis=1)
            self.item_feat.reset_index(inplace=True)
            self._fill_nan(self.item_feat)

        # load network features
        if self.config['network_feat_name'] is not None:
            self.network_feat = [None] * len(self.config['network_feat_name'])
            self.node_link = [None] * len(self.config['network_feat_name'])
            self.node_relink = [None] * len(self.config['network_feat_name'])
            self.mapped_fields = [[field.split(':')[0] if field != None else field for field in fields] for fields in self.config['mapped_feat_field']]
            for i, (name, fields) in enumerate(zip(self.config['network_feat_name'], self.config['network_feat_field'])):
                if len(name) == 2:
                    net_name, link_name = name
                    net_field, link_field = fields
                    link = self._load_feat(os.path.join(data_dir, link_name), self.config['network_feat_header'][i][1],
                                           field_sep, link_field, update_dict=False).to_numpy()
                    self.node_link[i] = dict(link)
                    self.node_relink[i] = dict(link[:, [1, 0]])
                    feat = self._load_feat(
                        os.path.join(data_dir, net_name), self.config['network_feat_header'][i][0], field_sep, net_field)
                    for j, col in enumerate(feat.columns):
                        if self.mapped_fields[i][j] != None:
                            feat[col] = [self.node_relink[i][id] if id in self.node_relink[i] else id for id in feat[col]]
                    self.network_feat[i] = feat
                else:
                    net_name, net_field = name[0], fields[0]
                    self.network_feat[i] = self._load_feat(
                        os.path.join(data_dir, net_name), self.config['network_feat_header'][i][0], field_sep, net_field)

    def _fill_nan(self, feat, mapped=False):
        r"""Fill the missing data in the original data.

        For token type, `[PAD]` token is used.
        For float type, the mean value is used.
        For token_seq type and float_seq, the empty numpy array is used.
        """
        for field in feat:
            ftype = self.field2type[field]
            if ftype == 'float':
                feat[field].fillna(value=feat[field].mean(), inplace=True)
            elif ftype == 'token':
                feat[field].fillna(value=0 if mapped else '[PAD]', inplace=True)
            elif ftype == 'token_seq':
                dtype = np.int64 if mapped else str
                feat[field] = \
                    feat[field].map(lambda x: np.array([], dtype=dtype) if isinstance(x, float) else x)
            elif ftype == 'float_seq':
                feat[field] = \
                    feat[field].map(lambda x: np.array([], dtype=np.float64) if isinstance(x, float) else x)
            else:
                raise ValueError(f'field type {ftype} is not supported. \
                    Only supports float, token, token_seq, float_seq.')

    def _load_feat(self, feat_path, header, sep, feat_cols, update_dict=True):
        r"""Load the feature from a given a feature file."""
        # fields, types_of_fields = zip(*( _.split(':') for _ in feat_cols))
        fields = []
        types_of_fields = []
        seq_seperators = {}
        for feat in feat_cols:
            s = feat.split(':')
            fields.append(s[0])
            types_of_fields.append(s[1])
            if len(s) == 3:
                seq_seperators[s[0]] = s[2].split('"')[1]

        dtype = [np.float64 if _ == 'float' else str for _ in types_of_fields]
        if update_dict:
            self.field2type.update(dict(zip(fields, types_of_fields)))

        if not "encoding_method" in self.config:
            self.config['encoding_method'] = 'utf-8'
        if self.config['encoding_method'] is None:
            self.config['encoding_method'] = 'utf-8'

        feat = pd.read_csv(feat_path, sep=sep, header=header, names=fields,
                           dtype=dict(zip(fields, dtype)), engine='python', index_col=False,
                           encoding=self.config['encoding_method'])[list(fields)]
        # seq_sep = self.config['seq_separator']
        for i, (col, t) in enumerate(zip(fields, types_of_fields)):
            if not t.endswith('seq'):
                if update_dict and (col not in self.field2maxlen):
                    self.field2maxlen[col] = 1
                continue
            feat[col].fillna(value='', inplace=True)
            cast = float if 'float' in t else str
            feat[col] = feat[col].map(
                lambda _: np.array(list(map(cast, filter(None, _.split(seq_seperators[col])))), dtype=cast)
            )
            if update_dict and (col not in self.field2maxlen):
                self.field2maxlen[col] = feat[col].map(len).max()
        return feat

    def _get_map_fields(self):
        #fields_share_space = self.config['fields_share_space'] or []
        if self.config['network_feat_name'] is not None:
            network_fields = {col: self.mapped_fields[i][j] for i, net in enumerate(self.network_feat) for j, col in enumerate(net.columns) if self.mapped_fields[i][j] != None}
        else:
            network_fields = {}
        fields_share_space = [[f] for f, t in self.field2type.items() if ('token' in t) and (f not in network_fields)]
        for k, v in network_fields.items():
            for field_set in fields_share_space:
                if v in field_set:
                    field_set.append(k)
        return fields_share_space

    def _get_feat_list(self):
        # if we have more features, please add here
        feat_list = [self.inter_feat, self.user_feat, self.item_feat]
        if self.config['network_feat_name'] is not None:
            feat_list.extend(self.network_feat)
        # return list(feat for feat in feat_list if feat is not None)
        return feat_list

    def _map_all_ids(self):
        r"""Map tokens to index."""
        fields_share_space = self._get_map_fields()
        feat_list = self._get_feat_list()
        for field_set in fields_share_space:
            flag = self.config['network_feat_name'] is not None \
                and (self.fuid in field_set or self.fiid in field_set)
            token_list = []
            field_feat = [(field, feat, idx) for field in field_set
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
            max_user_or_item_id = np.max(
                tid_list[:split_points[count_inter_user_or_item-1]]) + 1 if flag else 0
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
                            self.field2token2idx[field] = {
                                tokens[i]: i for i in range(max_user_or_item_id)}
                        else:
                            tokens_ori = self._get_ori_token(idx-3, tokens)
                            self.field2tokens[field] = tokens_ori
                            self.field2token2idx[field] = {
                                t: i for i, t in enumerate(tokens_ori)}
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
            self.user_feat = pd.DataFrame(
                {self.fuid: np.arange(self.num_users)})

        if self.item_feat is not None:
            self.item_feat.set_index(self.fiid, inplace=True)
            self.item_feat = self.item_feat.reindex(np.arange(self.num_items))
            self.item_feat.reset_index(inplace=True)
            self._fill_nan(self.item_feat, mapped=True)
        else:
            self.item_feat = pd.DataFrame(
                {self.fiid: np.arange(self.num_items)})

    def _post_preprocess(self):
        if self.ftime in self.inter_feat:
            if self.field2type[self.ftime] == 'str':
                assert 'time_format' in self.config, "time_format is required when timestamp is string."
                time_format = self.config['time_format']
                self.inter_feat[self.ftime] = pd.to_datetime(self.inter_feat[self.ftime], format=time_format)
            elif self.field2type[self.ftime] == 'float':
                pass
            else:
                raise ValueError(f'The field [{self.ftime}] should be float or str type')
        self._prepare_user_item_feat()

    def _recover_unmapped_feature(self, feat):
        feat = feat.copy()
        for field in feat:
            if field in self.field2tokens:
                feat[field] = feat[field].map(
                    lambda x: self.field2tokens[field][x])
        return feat

    def _drop_duplicated_pairs(self):
        # after drop, the interaction of user may be smaller than the min_user_inter, which will cause split problem
        # So we move the drop before filter to ensure after filtering, interactions of user and item are larger than min.
        first_item_idx = ~self.inter_feat.duplicated(
            subset=[self.fuid, self.fiid], keep='first')
        self.inter_feat = self.inter_feat[first_item_idx]

    def _filter(self, min_user_inter, min_item_inter):
        self._filter_ratings()
        item_list = self.inter_feat[self.fiid]
        item_idx_list, items = pd.factorize(item_list)
        user_list = self.inter_feat[self.fuid]
        user_idx_list, users = pd.factorize(user_list)
        user_item_mat = ssp.csc_matrix(
            (np.ones_like(user_idx_list), (user_idx_list, item_idx_list)))
        cols = np.arange(items.size)
        rows = np.arange(users.size)
        while(True): # TODO: only delete users/items in inter_feat, users/items in user/item_feat should also be deleted.
            m, n = user_item_mat.shape
            col_sum = np.squeeze(user_item_mat.sum(axis=0).A)
            col_ind = col_sum >= min_item_inter
            col_count = np.count_nonzero(col_ind)
            if col_count > 0:
                cols = cols[col_ind]
                user_item_mat = user_item_mat[:, col_ind]
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
                # @todo add output info if necessary

        keep_users = set(users[rows])
        keep_items = set(items[cols])
        keep = user_list.isin(keep_users)
        keep &= item_list.isin(keep_items)
        self.inter_feat = self.inter_feat[keep]
        self.inter_feat.reset_index(drop=True, inplace=True)
        if self.user_feat is not None:
           self.user_feat = self.user_feat[self.user_feat[self.fuid].isin(keep_users)]
           self.user_feat.reset_index(drop=True, inplace=True)
        if self.item_feat is not None:
           self.item_feat = self.item_feat[self.item_feat[self.fiid].isin(keep_items)]
           self.item_feat.reset_index(drop=True, inplace=True)

    def get_graph(self, idx, form='coo', value_fields=None, row_offset=0, col_offset=0, bidirectional=False, shape=None):
        """
        Returns a single graph or a graph composed of several networks. If more than one graph is passed into the methods, ``shape`` must be specified.

        Args:
            idx(int, list): the indices of the feat or networks. The index of ``inter_feat`` is set to ``0`` by default
            and the index of networks(such as knowledge graph and social network) is started by ``1`` corresponding to the dataset configuration file i.e. ``datasetname.yaml``.
            form(str): the form of the returned graph, can be 'coo', 'csr' or 'dgl'. Default: ``None``.
            value_fields(str, list): the value field in each graph. If value_field isn't ``None``, the values in this column will fill the adjacency matrix.
            row_offset(int, list): the offset of each row in corrresponding graph.
            col_offset(int, list): the offset of each column in corrresponding graph.
            bidirectional(bool, list): whether to turn the graph into bidirectional graph or not. Default: False
            shape(tuple): the shape of the returned graph. If more than one graph is passed into the methods, ``shape`` must be specified.

        Returns:
           graph(coo_matrix, csr_matrix or DGLGraph): a single graph or a graph composed of several networks in specified form.
           If the form is ``DGLGraph``, the relaiton type of the edges is stored in graph.edata['value'].
           num_relations(int): the number of relations in the combined graph.
           [ ['pad'], relation_0_0, relation_0_1, ..., relation_0_n, ['pad'], relation_1_0, relation_1_1, ..., relation_1_n]
        """
        if type(idx) == int:
            idx = [idx]
        if type(value_fields) == str or value_fields == None:
            value_fields = [value_fields] * len(idx)
        if type(bidirectional) == bool or bidirectional == None:
            bidirectional = [bidirectional] * len(idx)
        if type(row_offset) == int or row_offset == None:
            row_offset = [row_offset] * len(idx)
        if type(col_offset) == int or col_offset == None:
            col_offset = [col_offset] * len(idx)
        assert len(idx) == len(value_fields) and len(idx) == len(bidirectional)
        if shape is not None:
            assert type(shape) == list or type(shape) == tuple, 'the type of shape should be list or tuple'

        rows, cols, vals = [], [], []
        n, m, val_off = 0, 0, 0
        for id, value_field, bidirectional, row_off, col_off in zip(
                idx, value_fields, bidirectional, row_offset, col_offset):
            tmp_rows, tmp_cols, tmp_vals, val_off, tmp_n, tmp_m = self._get_one_graph(
                id, value_field, row_off, col_off, val_off, bidirectional)
            rows.append(tmp_rows)
            cols.append(tmp_cols)
            vals.append(tmp_vals)
            n += tmp_n
            m += tmp_m
        if shape == None or (type(shape) != tuple and type(shape) != list):
            if len(idx) > 1:
                raise ValueError(
                    f'If the length of idx is larger than 1, user should specify the shape of the combined graph.')
            else:
                shape = (n, m)
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        vals = torch.cat(vals)
        if form == 'coo':
            from scipy.sparse import coo_matrix
            return coo_matrix((vals, (rows, cols)), shape), val_off
        elif form == 'csr':
            from scipy.sparse import csr_matrix
            return csr_matrix((vals, (rows, cols)), shape), val_off
        elif form == 'dgl':
            import dgl
            assert shape[0] == shape[1], \
                'only support homogeneous graph in form of dgl, shape[0] must epuals to shape[1].'
            graph = dgl.graph((rows, cols), num_nodes=shape[0])
            graph.edata['value'] = vals
            return graph, val_off

    def _get_one_graph(self, id, value_field=None, row_offset=0, col_offset=0, val_offset=0, bidirectional=False):
        """
        Gets rows, cols and values in one graph.
        If several graphs are to be combined into one, offset should be added on the edge value in each graph to avoid conflict.
        Then the edge value will be: .. math:: offset + vals. (.. math:: offset + 1 in user-item graph). The offset will be reset to ``offset + len(self.field2tokens[value_field])`` in next graph.
        If bidirectional is True, the inverse edge values in the graph will be set to ``offset + corresponding_canonical_values + len(self.field2tokens[value_field]) - 1``.
        If all edges in the graph are sorted by their values in a list, the list will be:
            ['[PAD]', canonical_edge_1, canonical_edge_2, ..., canonical_edge_n, inverse_edge_1, inverse_edge_2, ..., inverse_edge_n]

        Args:
            id(int): the indix of the feat or network. The index of ``inter_feat`` is set to ``0`` by default
            and the index of networks(such as knowledge graph and social network) is started by ``1`` corresponding to the dataset configuration file i.e. ``datasetname.yaml``.
            value_field(str): the value field in the graph. If value_field isn't ``None``, the values in this column will fill the adjacency matrix.
            row_offset(int): the offset of the row in the graph. Default: 0.
            col_offset(int): the offset of the column in the graph. Default: 0.
            val_offset(int): the offset of the edge value in the graph. If several graphs are to be combined into one,
            offset should be added on the edge value in each graph to avoid conflict. Default: 0.
            bidirectional(bool): whether to turn the graph into bidirectional graph or not. Default: False

        Returns:
            rows(torch.Tensor): source nodes in all edges in the graph.
            cols(torch.Tensor): destination nodes in all edges in the graph.
            values(torch.Tensor): values of all edges in the graph.
            num_rows(int): number of source nodes.
            num_cols(int): number of destination nodes.
        """
        if id == 0:
            source_field = self.fuid
            target_field = self.fiid
            feat = self.inter_feat[self.inter_feat_subset]
        else:
            if self.network_feat is not None:
                if id - 1 < len(self.network_feat):
                    feat = self.network_feat[id - 1]
                    if len(feat.fields) == 2:
                        source_field, target_field = feat.fields[:2]
                    elif len(feat.fields) == 3:
                        source_field, target_field = feat.fields[0], feat.fields[2]
                else:
                    raise ValueError(
                        f'idx [{id}] is larger than the number of network features [{len(self.network_feat)}] minus 1')
            else:
                raise ValueError(
                    f'No network feature is input while idx [{id}] is larger than 1')
        if id == 0:
            source = feat[source_field] + row_offset
            target = feat[target_field] + col_offset
        else:
            source = feat.get_col(source_field) + row_offset
            target = feat.get_col(target_field) + col_offset
        if bidirectional:
            rows = torch.cat([source, target])
            cols = torch.cat([target, source])
        else:
            rows = source
            cols = target

        if value_field is not None:
            if id == 0 and value_field == 'inter':
                if bidirectional:
                    vals = torch.tensor(
                        [val_offset + 1] * len(source) + [val_offset + 2] * len(source))
                    val_offset += (1 + 2)
                else:
                    vals = torch.tensor([val_offset + 1] * len(source))
                    val_offset += (1 + 1)
            elif value_field in feat.fields:
                if bidirectional:
                    vals = feat.get_col(value_field) + val_offset
                    inv_vals = feat.get_col(
                        value_field) + len(self.field2tokens[value_field]) - 1 + val_offset
                    vals = torch.cat([vals, inv_vals])
                    val_offset += 2 * len(self.field2tokens[value_field]) - 1
                else:
                    vals = feat.get_col(value_field) + val_offset
                    val_offset += len(self.field2tokens[value_field])
            else:
                raise ValueError(
                    f'valued_field [{value_field}] does not exist')
        else:
            vals = torch.ones(len(rows))
        return rows, cols, vals, val_offset, self.num_values(source_field), self.num_values(target_field)

    def _split_by_ratio(self, ratio, data_count, user_mode):
        r"""Split dataset into train/valid/test by specific ratio."""
        m = len(data_count)
        if not user_mode:
            splits = np.outer(data_count, ratio).astype(np.int32)
            splits[:, 0] = data_count - splits[:, 1:].sum(axis=1)
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

        splits = np.hstack(
            [np.zeros((m, 1), dtype=np.int32), np.cumsum(splits, axis=1)])
        cumsum = np.hstack([[0], data_count.cumsum()[:-1]])
        splits = cumsum.reshape(-1, 1) + splits
        return splits, data_count.index if m > 1 else None

    def _split_by_num(self, num, data_count):
        r"""Split dataset into train/valid/test by specific ratio.
        num: list of int
        assert split_mode is entry                       
        """
        m = len(data_count)
        splits = np.hstack([0, num]).cumsum().reshape(1, -1)
        if splits[0][-1] == data_count.values.sum():
            return splits, data_count.index if m > 1 else None
        else:
            ValueError(f'Expecting the number of interactions \
            should be equal to the sum of {num}')
    
    def _split_by_leave_one_out(self, leave_one_num, data_count, rep=True):
        r"""Split dataset into train/valid/test by leave one out method.
        The split methods are usually used for sequential recommendation, where the last item of the item sequence will be used for test.

        Args:
            leave_one_num(int): the last ``leave_one_num`` items of the sequence will be splited out.
            data_count(pandas.DataFrame or numpy.ndarray):  entry range for each user or number of all entries.
            rep(bool, optional): whether there should be repititive items in the sequence.
        """
        m = len(data_count)
        cumsum = data_count.cumsum()[:-1]
        if rep:
            splits = np.ones((m, leave_one_num + 1), dtype=np.int32)
            splits[:, 0] = data_count - leave_one_num
            for _ in range(leave_one_num):
                idx = splits[:, 0] < 1
                splits[idx, 0] += 1
                splits[idx, _] -= 1
            splits = np.hstack(
                [np.zeros((m, 1), dtype=np.int32), np.cumsum(splits, axis=1)])
        else:
            def get_splits(bool_index):
                idx = bool_index.values.nonzero()[0]
                if len(idx) > 2:
                    return [0, idx[-2], idx[-1], len(idx)]
                elif len(idx) == 2:
                    return [0, idx[-1], idx[-1], len(idx)]
                else:
                    return [0, len(idx), len(idx), len(idx)]
            splits = np.array([get_splits(bool_index)
                              for bool_index in np.split(self.first_item_idx, cumsum)])

        cumsum = np.hstack([[0], cumsum])
        splits = cumsum.reshape(-1, 1) + splits
        return splits, data_count.index if m > 1 else None

    def _get_data_idx(self, splits):
        r""" Return data index for train/valid/test dataset.
        """
        splits, uids = splits
        data_idx = [list(zip(splits[:, i-1], splits[:, i]))
                    for i in range(1, splits.shape[1])]
        if not getattr(self, 'fmeval', False):
            if uids is not None:
                d = [torch.from_numpy(np.hstack([np.arange(*e) for e in data_idx[0]]))]
                for _ in data_idx[1:]:
                    d.append(torch.tensor([[u, *e] for u, e in zip(uids, _) if e[1] > e[0]])) # skip users who don't have interactions in valid or test dataset.
                return d
            else:
                d = [torch.from_numpy(np.hstack([np.arange(*e)
                                      for e in data_idx[0]]))]
                for _ in data_idx[1:]:
                    start, end = _[0]
                    data = self.inter_feat.get_col(self.fuid)[start:end]
                    uids, counts = data.unique_consecutive(return_counts=True)
                    cumsum = torch.hstack(
                        [torch.tensor([0]), counts.cumsum(-1)]) + start
                    d.append(torch.tensor(
                        [[u, st, en] for u, st, en in zip(uids, cumsum[:-1], cumsum[1:])]))
                return d
        else:
            return [torch.from_numpy(np.hstack([np.arange(*e) for e in _])) for _ in data_idx]

    def __len__(self):
        r"""Return the length of the dataset."""
        return len(self.data_index)

    def _get_pos_data(self, index):
        if self.data_index.dim() > 1:
            idx = self.data_index[index]
            data = {self.fuid: idx[:, 0]}
            data.update(self.user_feat[data[self.fuid]])
            start = idx[:, 1]
            end = idx[:, 2]
            lens = end - start
            l = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
            d = self.inter_feat.get_col(self.fiid)[l]
            rating = self.inter_feat.get_col(self.frating)[l]
            data[self.fiid] = pad_sequence(
                d.split(tuple(lens.numpy())), batch_first=True)
            data[self.frating] = pad_sequence(
                rating.split(tuple(lens.numpy())), batch_first=True)
        else:
            idx = self.data_index[index]
            data = self.inter_feat[idx]
            uid, iid = data[self.fuid], data[self.fiid]
            data.update(self.user_feat[uid])
            data.update(self.item_feat[iid])

        if 'user_hist' in data:
            user_count = self.user_count[data[self.fuid]].max()
            data['user_hist'] = data['user_hist'][:, 0:user_count]

        return data


    def _get_neg_data(self, data: Dict):
        if 'user_hist' not in data:
            user_count = self.user_count[data[self.fuid]].max()
            user_hist = self.user_hist[data[self.fuid]][:, 0:user_count]
        else:
            user_hist = data['user_hist']
        neg_id = uniform_sampling(data[self.frating.size(0)], self.num_items,
                                    self.neg_count, user_hist).long()   # [B, neg]
        neg_id = neg_id.transpose(0,1).contiguous().view(-1)    # [neg*B]
        neg_item_feat = self.item_feat[neg_id]
        # negatives should be flatten here.
        # After flatten and concat, the batch size will be B*(1+neg)
        for k, v in data.items():
            if k in neg_item_feat:
                data[k] = torch.cat([v, neg_item_feat[k]], dim=0)
            elif k != self.frating:
                data[k] = v.tile((self.neg_count+1,))
            else:   # rating
                neg_rating = torch.zeros_like(neg_id)
                data[k] = torch.cat((v, neg_rating), dim=0)
        return data

    def __getitem__(self, index):
        r"""Get data at specific index.

        Args:
            index(int): The data index.
        Returns:
            dict: A dict contains different feature.
        """
        data = self._get_pos_data(index)
        if self.eval_mode and 'user_hist' not in data:
            user_count = self.user_count[data[self.fuid]].max()
            data['user_hist'] = self.user_hist[data[self.fuid]][:, 0:user_count]
        else:
            # Negative sampling in dataset.
            # Only uniform sampling is supported now.
            if getattr(self, 'neg_count', None) is not None:
                data = self._get_neg_data(data)
        return data


    def _copy(self, idx):
        d = copy.copy(self)
        d.data_index = idx
        return d

    def _init_sampler(self, dataset_sampler, dataset_neg_count):
        self.neg_count = dataset_neg_count
        self.sampler = dataset_sampler
        if self.sampler is not None:
            assert self.sampler == 'uniform', "`dataset_sampler` only support uniform sampler now."
            assert self.neg_count is not None, "`dataset_neg_count` are required when `dataset_sampler` is used."
            self.logger.warning("The rating of the sampled negatives will be set as 0.")
            if not self.config['drop_low_rating']:
                self.logger.warning("Please attention the `drop_low_rating` is False and "
                                    "the dataset is a rating dataset, the sampled negatives will "
                                    "be treated as interactions with rating 0.")
            self.logger.warning(f"With the sampled negatives, the batch size will be "
                                f"{self.neg_count+1} times as the batch size set in the "
                                f"configuration file. For example, `batch_size=16` and "
                                f"`dataset_neg_count=2` will load batches with size 48.")

    def build(
            self,
            split_ratio: List = [0.8, 0.1, 0.1],
            shuffle: bool = True,
            split_mode: str = 'user_entry',
            fmeval: bool = False,
            dataset_sampler: str = None,
            dataset_neg_count: int = None,
            **kwargs
        ):
        """Build dataset.

        Args:
            split_ratio(numeric): split ratio for data preparition. If given list of float, the dataset will be splited by ratio. If given a integer, leave-n method will be used.

            shuffle(bool, optional): set True to reshuffle the whole dataset each epoch. Default: ``True``

            split_mode(str, optional): controls the split mode. If set to ``user_entry``, then the interactions of each user will be splited into 3 cut.
            If ``entry``, then dataset is splited by interactions. If ``user``, all the users will be splited into 3 cut. Default: ``user_entry``

            fmeval(bool, optional): set True for MFDataset and ALSDataset when use TowerFreeRecommender. Default: ``False``

        Returns:
            list: A list contains train/valid/test data-[train, valid, test]
        """
        self.fmeval = fmeval
        self.split_mode = split_mode
        self._init_sampler(dataset_sampler, dataset_neg_count)
        return self._build(split_ratio, shuffle, split_mode, False)

    def _build(self, ratio_or_num, shuffle, split_mode, rep):
        # for general recommendation, only support non-repetive recommendation
        # keep first data, sorted by time or not, split by user or not
        if not hasattr(self, 'first_item_idx'):
            self.first_item_idx = ~self.inter_feat.duplicated(
                subset=[self.fuid, self.fiid], keep='first')
        if self.drop_dup:
            self.inter_feat = self.inter_feat[self.first_item_idx]

        if (split_mode == 'user_entry') or (split_mode == 'user'):
            if self.ftime in self.inter_feat:
                self.inter_feat.sort_values(by=[self.fuid, self.ftime], inplace=True)
                self.inter_feat.reset_index(drop=True, inplace=True)
            else:
                self.inter_feat.sort_values(by=self.fuid, inplace=True)
                self.inter_feat.reset_index(drop=True, inplace=True)

        if split_mode == 'user_entry':
            user_count = self.inter_feat[self.fuid].groupby(
                self.inter_feat[self.fuid], sort=False).count()
            if shuffle:
                cumsum = np.hstack([[0], user_count.cumsum()[:-1]])
                idx = np.concatenate([np.random.permutation(c) + start
                    for start, c in zip(cumsum, user_count)])
                self.inter_feat = self.inter_feat.iloc[idx].reset_index(drop=True)
        elif split_mode == 'entry':
            if isinstance(ratio_or_num, list) and \
                isinstance(ratio_or_num[0], int): # split by num
                user_count = self.inter_feat[self.fuid].groupby(
                self.inter_feat[self.fuid], sort=True).count()
            else:
                if shuffle:
                    self.inter_feat = self.inter_feat.sample(
                        frac=1).reset_index(drop=True)
                user_count = np.array([len(self.inter_feat)])
        elif split_mode == 'user':
            user_count = self.inter_feat[self.fuid].groupby(
                self.inter_feat[self.fuid], sort=False).count()

        if isinstance(ratio_or_num, int):
            splits = self._split_by_leave_one_out(
                ratio_or_num, user_count, rep)
        elif isinstance(ratio_or_num, list) and \
            isinstance(ratio_or_num[0], float):
            splits = self._split_by_ratio(
                ratio_or_num, user_count, split_mode == 'user')
        else:
            splits = self._split_by_num(
                ratio_or_num, user_count)
            
        splits_ = splits[0][0]
        if split_mode == 'entry':
            if isinstance(self, AEDataset) or isinstance(self, SeqDataset):
                ucnts = pd.DataFrame({self.fuid : splits[1]})
                for i, (start, end) in enumerate(zip(splits_[:-1], splits_[1:])):
                    self.inter_feat[start:end] = self.inter_feat[start:end].sort_values(
                        by=[self.fuid, self.ftime] if self.ftime in self.inter_feat 
                        else self.fuid)
                    ucnts[i] = self.inter_feat[start:end][self.fuid].groupby(
                        self.inter_feat[self.fuid], sort=True).count().values
                self.inter_feat.sort_values(by=[self.fuid], inplace=True, kind='mergesort')
                self.inter_feat.reset_index(drop=True, inplace=True)
                ucnts = ucnts.astype(int)
                ucnts = torch.from_numpy(ucnts.values)
                u_cumsum = ucnts[:, 1:].cumsum(dim=1)
                u_start = torch.hstack(
                    [torch.tensor(0), u_cumsum[:, -1][:-1]]).view(-1, 1).cumsum(dim=0)
                splits = torch.hstack([u_start, u_cumsum + u_start])
                uids = ucnts[:, 0]
                if isinstance(self, AEDataset):
                    splits = (splits, uids.view(-1, 1))
                else:
                    splits = (splits.numpy(), uids)
            else:
                for start, end in zip(splits_[:-1], splits_[1:]):
                    self.inter_feat[start:end] = self.inter_feat[start:end].sort_values(
                        by=[self.fuid, self.ftime] if self.ftime in self.inter_feat 
                        else self.fuid)
        

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
        r"""Convert the data type from TensorFrame to Tensor
        """
        self.inter_feat = TensorFrame.fromPandasDF(self.inter_feat, self)
        self.user_feat = TensorFrame.fromPandasDF(self.user_feat, self)
        self.item_feat = TensorFrame.fromPandasDF(self.item_feat, self)
        if hasattr(self, 'network_feat'):
            for i in range(len(self.network_feat)):
                self.network_feat[i] = TensorFrame.fromPandasDF(
                    self.network_feat[i], self)

    def train_loader(self, batch_size, shuffle=True, num_workers=1, drop_last=False, ddp=False):
        r"""Return a dataloader for training.

        Args:
            batch_size(int): the batch size for training data.

            shuffle(bool,optimal): set to True to have the data reshuffled at every epoch. Default:``True``.

            num_workers(int, optimal): how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process. (default: ``1``)

            drop_last(bool, optimal): set to True to drop the last mini-batch if the size is smaller than given batch size. Default: ``False``

            load_combine(bool, optimal): set to True to combine multiple loaders as :doc:`ChainedDataLoader <chaineddataloader>`. Default: ``False``

        Returns:
            list or ChainedDataLoader: list of loaders if load_combine is True else ChainedDataLoader.

        .. note::
            Due to that index is used to shuffle the dataset and the data keeps remained, `num_workers > 0` may get slower speed.
        """
        self.eval_mode = False # set mode to training.
        return self.loader(batch_size, shuffle, num_workers, drop_last, ddp)

    def loader(self, batch_size, shuffle=True, num_workers=1, drop_last=False, ddp=False):
        # if not ddp:
        if self.data_index.dim() > 1:  # has sample_length
            sampler = SortedDataSampler(self, batch_size, shuffle, drop_last)
        else:
            sampler = DataSampler(self, batch_size, shuffle, drop_last)

        if ddp:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False)

        output = DataLoader(self, sampler=sampler, batch_size=None,
                            shuffle=False, num_workers=num_workers,
                            persistent_workers=False)

        # if ddp:
        #     sampler = torch.utils.data.distributed.DistributedSampler(self, shuffle=shuffle, drop_last=drop_last)
        #     output = DataLoader(self, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
        return output

    @property
    def sample_length(self):
        if self.data_index.dim() > 1:
            return self.data_index[:, 2] - self.data_index[:, 1]
        else:
            raise ValueError('can not compute sample length for this dataset')

    def eval_loader(self, batch_size, num_workers=1, ddp=False):
        self.eval_mode = True
        if not getattr(self, 'fmeval', False):
            # if ddp:
            #     sampler = torch.utils.data.distributed.DistributedSampler(self, shuffle=False)
            #     output = DataLoader(
            #         self, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
            # else:
            sampler = SortedDataSampler(self, batch_size)
            if ddp:
                sampler = DistributedSamplerWrapper(sampler, shuffle=False)
            output = DataLoader(
                self, sampler=sampler, batch_size=None, shuffle=False,
                num_workers=num_workers, persistent_workers=False)
            return output
        else:
            self.eval_mode = True
            return self.loader(batch_size, shuffle=False, num_workers=num_workers, ddp=ddp)

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
        r"""Get user or item interaction history.

        Args:
            isUser(bool, optional): Default: ``True``.

        Returns:
            torch.Tensor: padded user or item hisoty.

            torch.Tensor: length of the history sequence.
        """
        user_array = self.inter_feat.get_col(self.fuid)[self.inter_feat_subset]
        item_array = self.inter_feat.get_col(self.fiid)[self.inter_feat_subset]
        sorted, index = torch.sort(user_array if isUser else item_array)
        user_item, count = torch.unique_consecutive(sorted, return_counts=True)
        list_ = torch.split(
            item_array[index] if isUser else user_array[index], tuple(count.numpy()))
        tensors = [torch.tensor([], dtype=torch.int64) for _ in range(
            self.num_users if isUser else self.num_items)]
        for i, l in zip(user_item, list_):
            tensors[i] = l
        user_count = torch.tensor([len(e) for e in tensors])
        tensors = pad_sequence(tensors, batch_first=True)
        return tensors, user_count

    def get_network_field(self, network_id, feat_id, field_id):
        """
        Returns the specified field name in some network.
        For example, if the head id field is in the first feat of KG network and is the first column of the feat and the index of KG network is 1.
        To get the head id field, the method can be called like this ``train_data.get_network_field(1, 0, 0)``.

        Args:
            network_id(int) : the index of network corresponding to the dataset configuration file.
            feat_id(int): the index of the feat in the network.
            field_id(int): the index of the wanted field in above feat.

        Returns:
            field(str): the wanted field.
        """
        return self.config['network_feat_field'][network_id][feat_id][field_id].split(':')[0]

    @property
    def inter_feat_subset(self):
        r""" Data index.
        """
        if self.data_index.dim() > 1:
            return torch.cat([torch.arange(s, e) for s, e in zip(self.data_index[:, 1], self.data_index[:, 2])])
        else:
            return self.data_index

    @property
    def item_freq(self):
        r""" Item frequency (or popularity).

        Returns:
            torch.Tensor: ``[num_items,]``. The times of each item appears in the dataset.
        """
        if not hasattr(self, 'data_index'):
            raise ValueError(
                'please build the dataset first by call the build method')
        l = self.inter_feat.get_col(self.fiid)[self.inter_feat_subset]
        it, count = torch.unique(l, return_counts=True)
        it_freq = torch.zeros(self.num_items, dtype=torch.int64)
        it_freq[it] = count
        return it_freq

    @property
    def num_users(self):
        r"""Number of users.

        Returns:
            int: number of users.
        """
        return self.num_values(self.fuid)

    @property
    def num_items(self):
        r"""Number of items.

        Returns:
            int: number of items.
        """
        return self.num_values(self.fiid)

    @property
    def num_inters(self):
        r"""Number of total interaction numbers.

        Returns:
            int: number of interactions in the dataset.
        """
        return len(self.inter_feat)

    def num_values(self, field):
        r"""Return number of values in specific field.

        Args:
            field(str): the field to be counted.

        Returns:
            int: number of values in the field.

        .. note::
            This method is used to return ``num_items``, ``num_users`` and ``num_inters``.
        """
        if 'token' not in self.field2type[field]:
            return self.field2maxlen[field]
        else:
            return len(self.field2tokens[field])


class AEDataset(MFDataset):
    def build(
            self,
            split_ratio=[0.8, 0.1, 0.1],
            shuffle=False,
            split_mode='user_entry',
            dataset_sampler=None, 
            dataset_neg_count=None, 
            **kwargs
        ):
        """Build dataset.

        Args:
            ratio_or_num(numeric): split ratio for data preparition. If given list of float, the dataset will be splited by ratio. If given a integer, leave-n method will be used.

            shuffle(bool, optional): set True to reshuffle the whole dataset each epoch. Default: ``True``

            split_mode(str, optional): controls the split mode. If set to ``user_entry``, then the interactions of each user will be splited into 3 cut.
            If ``entry``, then dataset is splited by interactions. If ``user``, all the users will be splited into 3 cut. Default: ``user_entry``

            fmeval(bool, optional): set True for MFDataset and ALSDataset when use TowerFreeRecommender. Default: ``False``

        Returns:
            list or ChainedDataLoader: list of loaders if load_combine is True else ChainedDataLoader.
        """
        self.split_mode = split_mode
        self._init_sampler(dataset_sampler, dataset_neg_count)
        if split_mode == 'entry':
            # False if split_by_num
            shuffle = shuffle and \
                        not (isinstance(split_ratio, list) and \
                        isinstance(split_ratio[0], int))
        return self._build(split_ratio, shuffle, split_mode, False)

    def _get_data_idx(self, splits):
        splits, uids = splits
        # filter out users whose behaviors are not in valid and test data,
        # otherwise it will cause nan in metric calculation such as recall.
        # usually the reason is that the number of behavior is too small due to the sparsity.
        if self.split_mode == 'user_entry':
            mask = splits[:, 1] < splits[:, 2]
            splits, uids = splits[mask], uids[mask]
            data_idx = [list(zip(splits[:, i-1], splits[:, i]))
                        for i in range(1, splits.shape[1])]
            data_idx = [torch.tensor([[u, *e] for e, u in zip(_, uids)])
                        for _ in data_idx]
            data = [torch.cat((data_idx[0], data_idx[i]), -1)
                    for i in range(len(data_idx))]
        elif self.split_mode == 'entry':
            data_idx = [torch.hstack([uids, splits[:,i:i+2]])
                        for i in range(splits.shape[1] - 1)]
            data = [torch.cat((data_idx[0], data_idx[i]), -1)
                    for i in range(len(data_idx))]
            
        return data

    def __getitem__(self, index):
        idx = self.data_index[index]
        data = {self.fuid: idx[:, 0]}
        data.update(self.user_feat[data[self.fuid]])
        for i, n in enumerate(['in_', '']):
            start = idx[:, i * 3 + 1]
            end = idx[:, i * 3 + 2]
            lens = end - start
            l = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
            d = self.inter_feat[l]
            for k in d:
                d[k] = pad_sequence(d[k].split(
                    tuple(lens.numpy())), batch_first=True)
            d.update(self.item_feat[d[self.fiid]])
            for k, v in d.items():
                if k != self.fuid:
                    data[n+k] = v

        if self.eval_mode and 'user_hist' not in data:
            data['user_hist'] = data['in_'+self.fiid]
        else:
            if self.neg_count is not None:
                data = self._get_neg_data(data)
        return data

    @property
    def inter_feat_subset(self):
        index = torch.cat([torch.arange(s, e) for s, e in zip(
            self.data_index[:, -2], self.data_index[:, -1])])
        return index


class SeqDataset(MFDataset):
    @property
    def drop_dup(self):
        return False

    def build(
            self, 
            split_ratio=2, 
            split_mode='user_entry',
            rep=True, 
            train_rep=True, 
            dataset_sampler=None, 
            dataset_neg_count=None, 
            **kwargs
        ):
        self.test_rep = rep
        self.train_rep = train_rep if not rep else True
        self.split_mode = split_mode
        self._init_sampler(dataset_sampler, dataset_neg_count)

        return self._build(split_ratio, False, split_mode, rep) #TODO: add split method 'user'

    def _get_data_idx(self, splits):
        splits, uids = splits
        maxlen = self.config['max_seq_len'] or (splits[:, -1] - splits[:, 0]).max()

        def keep_first_item(dix, part):
            if ((dix == 0) and self.train_rep) or ((dix > 0) and self.test_rep):
                return part
            else:
                return part[self.first_item_idx.iloc[part[:, -1]].values]

        def get_slice(sp, u):
            data = np.array([[u, max(sp[0], i - maxlen), i]
                            for i in range(sp[0], sp[-1])], dtype=np.int64)
            sp -= sp[0]
            return np.split(data[1:], sp[1:-1]-1)
        output = [get_slice(sp, u) for sp, u in zip(splits, uids)]
        output = [torch.from_numpy(np.concatenate(_)) for _ in zip(*output)] # [[user, start, end]]
        output = [keep_first_item(dix, _) for dix, _ in enumerate(output)]
        return output

    def _get_pos_data(self, index):
        idx = self.data_index[index]
        data = {self.fuid: idx[:, 0]}
        data.update(self.user_feat[data[self.fuid]])
        target_data = self.inter_feat[idx[:, 2]]
        target_data.update(self.item_feat[target_data[self.fiid]])
        start = idx[:, 1]
        end = idx[:, 2]
        lens = end - start
        data['seqlen'] = lens
        l = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
        source_data = self.inter_feat[l]
        for k in source_data:
            source_data[k] = pad_sequence(source_data[k].split(
                tuple(lens.numpy())), batch_first=True)
        source_data.update(self.item_feat[source_data[self.fiid]])

        for n, d in zip(['in_', ''], [source_data, target_data]):
            for k, v in d.items():
                if k != self.fuid:
                    data[n+k] = v
        return data

    @property
    def inter_feat_subset(self):
        return self.data_index[:, -1]


class FullSeqDataset(SeqDataset):
    def _get_data_idx(self, splits):
        splits, uids = splits
        maxlen = self.config['max_seq_len'] or (splits[:, -1] - splits[:, 0]).max()

        def get_slice(sp, u):
            sp[1:] = sp[1:] - 1
            data = [np.array([[u, max(sp[0], sp[1]-maxlen), sp[1]]])]
            data += [np.array([[u, max(s-maxlen, sp[0]), s]]) for s in sp[2:]]
            return data
        output = [get_slice(sp, u) for sp, u in zip(splits, uids)]
        output = [torch.from_numpy(np.concatenate(_)) for _ in zip(*output)]
        return output


class SeqToSeqDataset(SeqDataset):

    def _get_data_idx(self, splits):
        # bug to fix : "user" split mode
        # split: [start, train_end, valid_end, test_end]
        splits, uids = splits
        maxlen = self.config['max_seq_len'] or (splits[:, -1] - splits[:, 0]).max()

        def keep_first_item(dix, part):
            # self.drop_dup is set to False in SeqDataset
            if ((dix == 0) and self.train_rep) or ((dix > 0) and self.test_rep):
                return part
            else:
                return part[self.first_item_idx.iloc[part[:, -1]].values]

        def get_slice(sp, u):
            # the length of the train slice should be maxlen + 1 to get train data with length maxlen
            data = [np.array([[u, max(sp[0], i - 1 - maxlen), i - 1]]) if (i - 1) > (max(sp[0], i - 1 - maxlen)) \
                else np.array([], dtype=np.int64).reshape((0, 3)) for i in sp[1:]]
            return tuple(data)

        output = [get_slice(sp, u) for sp, u in zip(splits, uids)]
        output = [torch.from_numpy(np.concatenate(_)) for _ in zip(*output)]
        output = [keep_first_item(dix, _) for dix, _ in enumerate(output)] # figure out
        return output

    def _get_pos_data(self, index):
        # training:
        # source: interval [idx[:, 1], idx[:, 2] - 1]
        # target: interval [idx[:, 1] + 1, idx[:, 2]]
        # valid/test:
        # source: interval [idx[:, 1], idx[:, 2] - 1]
        # target: idx[:, 2]
        idx = self.data_index[index]
        data = {self.fuid: idx[:, 0]}
        data.update(self.user_feat[data[self.fuid]])
        start = idx[:, 1]
        end = idx[:, 2]
        lens = end - start
        data['seqlen'] = lens
        l_source = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])
        # source_data
        source_data = self.inter_feat[l_source]
        for k in source_data:
            source_data[k] = pad_sequence(source_data[k].split(
                tuple(lens.numpy())), batch_first=True)
        source_data.update(self.item_feat[source_data[self.fiid]])
        # target_data
        if not self.eval_mode:
            l_target = l_source + 1
            target_data = self.inter_feat[l_target]
            for k in target_data:
                target_data[k] = pad_sequence(target_data[k].split(
                    tuple(lens.numpy())), batch_first=True)
            target_data.update(self.item_feat[target_data[self.fiid]])
        else:
            target_data = self.inter_feat[idx[:, 2]]
            target_data.update(self.item_feat[target_data[self.fiid]])

        for n, d in zip(['in_', ''], [source_data, target_data]):
            for k, v in d.items():
                if k != self.fuid:
                    data[n+k] = v
        return data

    @property
    def inter_feat_subset(self):
        """self.data_index : [num_users, 3]
        The intervel in data_index is both closed.
        data_index only includes interactions in the truncated sequence of a user, instead of all interactions.
        Return:
            torch.tensor: the history index in inter_feat. shape: [num_interactions_in_train]
        """
        start = self.data_index[:, 1]
        end = self.data_index[:, 2]
        return torch.cat([torch.arange(s, e + 1, dtype=s.dtype) for s, e in zip(start, end)], dim=0)

    def loader(self, batch_size, shuffle=True, num_workers=1, drop_last=False, ddp=False):
        # if not ddp:
        # Don't use SortedSampler here, it may hurt the performence of the model.
        sampler = DataSampler(self, batch_size, shuffle, drop_last)
        if ddp:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False)

        output = DataLoader(self, sampler=sampler, batch_size=None,
                            shuffle=False, num_workers=num_workers,
                            persistent_workers=False)
        return output


class TensorFrame(Dataset):
    r"""The main data structure used to save interaction data in RecStudio dataset.

    TensorFrame class can be regarded as one enhanced dict, which contains several fields of data (like: ``user_id``, ``item_id``, ``rating`` and so on).
    And TensorFrame have some useful strengths:

    - Generated from pandas.DataFrame directly.

    - Easy to get/add/remove fields.

    - Easy to get each interaction information.

    - Compatible for torch.utils.data.DataLoader, which provides a loader method to return batch data.
    """
    @classmethod
    def fromPandasDF(cls, dataframe, dataset):
        r"""Get a TensorFrame from a pandas.DataFrame.

        Args:
            dataframe(pandas.DataFrame): Dataframe read from csv file.
            dataset(recstudio.data.MFDataset): target dataset where the TensorFrame is used.

        Return:
            recstudio.data.TensorFrame: the TensorFrame get from the dataframe.
        """
        data = {}
        fields = []
        length = len(dataframe.index)
        for field in dataframe:
            fields.append(field)
            ftype = dataset.field2type[field]
            value = dataframe[field]
            if ftype == 'token_seq':
                seq_data = [torch.from_numpy(
                    d[:dataset.field2maxlen[field]]) for d in value]
                data[field] = pad_sequence(seq_data, batch_first=True)
            elif ftype == 'float_seq':
                seq_data = [torch.from_numpy(
                    d[:dataset.field2maxlen[field]]) for d in value]
                data[field] = pad_sequence(seq_data, batch_first=True)
            elif ftype == 'token':
                data[field] = torch.from_numpy(
                    dataframe[field].to_numpy(np.int64))
            else:
                data[field] = torch.from_numpy(
                    dataframe[field].to_numpy(np.float32))
        return cls(data, length, fields)

    def __init__(self, data, length, fields):
        self.data = data
        self.length = length
        self.fields = fields

    def get_col(self, field):
        r"""Get data from the specific field.

        Args:
            field(str): field name.

        Returns:
            torch.Tensor: data of corresponding filed.
        """
        return self.data[field]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ret = {}
        for field, value in self.data.items():
            ret[field] = value[idx]
        return ret

    def del_fields(self, keep_fields):
        r"""Delete fields that are *not in* ``keep_fields``.

        Args:
            keep_fields(list[str],set[str] or dict[str]): the fields need to remain.
        """
        fields = copy.deepcopy(self.fields)
        for f in fields:
            if f not in keep_fields:
                self.fields.remove(f)
                del self.data[f]

    def loader(self, batch_size, shuffle=False, num_workers=1, drop_last=False):
        r"""Create dataloader.

        Args:
            batch_size(int): batch size for mini batch.

            shuffle(bool, optional): whether to shuffle the whole data. (default `False`).

            num_workers(int, optional): how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process. (default: `1`).

            drop_last(bool, optinal): whether to drop the last mini batch when the size is smaller than the `batch_size`.

        Returns:
            torch.utils.data.DataLoader: the dataloader used to load all the data in the TensorFrame.
        """
        sampler = DataSampler(self, batch_size, shuffle, drop_last)
        output = DataLoader(self, sampler=sampler, batch_size=None,
                            shuffle=False, num_workers=num_workers,
                            persistent_workers=False)
        return output

    def add_field(self, field, value):
        r"""Add field to the TensorFrame.

        Args:
            field(str): the field name to be added.

            value(torch.Tensor): the value of the field.
        """
        self.data[field] = value

    def reindex(self, idx):
        r"""Shuffle the data according to the given `idx`.

        Args:
            idx(numpy.ndarray): the given data index.

        Returns:
            recstudio.data.TensorFrame: a copy of the TensorFrame after reindexing.
        """
        output = copy.deepcopy(self)
        for f in output.fields:
            output.data[f] = output.data[f][idx]
        return output


class DataSampler(Sampler):
    r"""Data sampler to return index for batch data.

    The datasampler generate batches of index in the `data_source`, which can be used in dataloader to sample data.

    Args:
        data_source(Sized): the dataset, which is required to have length.

        batch_size(int): batch size for each mini batch.

        shuffle(bool, optional): whether to shuffle the dataset each epoch. (default: `True`)

        drop_last(bool, optional): whether to drop the last mini batch when the size is smaller than the `batch_size`.(default: `False`)

        generator(optinal): generator to generate rand numbers. (default: `None`)
    """

    def __init__(self, data_source: Sized, batch_size, shuffle=True, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        if self.shuffle:
            output = torch.randperm(
                n, generator=generator).split(self.batch_size)
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
    r"""Data sampler to return index for batch data, aiming to collect data with similar lengths into one batch.

    In order to save memory in training producure, the data sampler collect data point with similar length into one batch.

    For example, in sequential recommendation, the interacted item sequence of different users may vary differently, which may cause
    a lot of padding. By considering the length of each sequence, gathering those sequence with similar lengths in the same batch can
    tackle the problem.

    If `shuffle` is `True`, length of sequence and the random index are combined together to reduce padding without randomness.

    Args:
        data_source(Sized): the dataset, which is required to have length.

        batch_size(int): batch size for each mini batch.

        shuffle(bool, optional): whether to shuffle the dataset each epoch. (default: `True`)

        drop_last(bool, optional): whether to drop the last mini batch when the size is smaller than the `batch_size`.(default: `False`)

        generator(optinal): generator to generate rand numbers. (default: `None`)
    """

    def __init__(self, data_source: Sized, batch_size, shuffle=False, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            output = torch.div(torch.randperm(n), (self.batch_size * 10), rounding_mode='floor')
            output = self.data_source.sample_length + output * \
                (self.data_source.sample_length.max() + 1)
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
    r"""ChainedDataLoader aims to combine several loaders in a chain.

    In some cases, several different dataloaders are used for one algorithm.

    Args:
        loaders(list[torch.utils.data.DataLoader]): list of dataloaders.

        nepoch(list or numpy.ndarray, optional): list with the same length as loaders, controls how many epochs each dataloader iterates for. (default: `None`)
    """

    def __init__(self, loaders, nepoch=None) -> None:
        self.loaders = loaders
        self.epoch = -1
        nepoch = np.ones(len(loaders)) if nepoch is None else np.array(nepoch)
        self.iter_idx = np.concatenate(
            [np.repeat(i, c) for i, c in enumerate(nepoch)])

    def __iter__(self):
        self.epoch += 1
        return iter(self.loaders[self.iter_idx[self.epoch % len(self.iter_idx)]])


class CombinedLoaders(object):
    def __init__(self, loaders) -> None:
        r"""
        The first loader is the main loader.
        """
        self.loaders = loaders

    def __len__(self):
        return len(self.loaders[0])

    def __iter__(self):
        for i, l in enumerate(self.loaders):
            self.loaders[i] = iter(l)
        return self

    def __next__(self):
        batch = next(self.loaders[0])
        for i, l in enumerate(self.loaders[1:]):
            try:
                batch.update(next(l))
            except StopIteration:
                self.loaders[i+1] = iter(self.loaders[i+1])
                batch.update(next(self.loaders[i+1]))
        return batch


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
