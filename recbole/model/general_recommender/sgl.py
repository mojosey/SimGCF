# -*- coding: utf-8 -*-
# @Time   : 2021/10/12
# @Author : Tian Zhen
# @Email  : chenyuwuxinn@gmail.com

r"""
SGL
################################################
Reference:
    Jiancan Wu et al. "SGL: Self-supervised Graph Learning for Recommendation" in SIGIR 2021.

Reference code:
    https://github.com/wujcan/SGL
"""

import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F

from torch_sparse import SparseTensor, matmul
from .PolyConv import PolyConvFrame, JacobiConv,PowerConv,low_filter,high_filter,fit_filter
from functools import partial



class SGL(GeneralRecommender):
    r"""SGL is a GCN-based recommender model.

    SGL supplements the classical supervised task of recommendation with an auxiliary
    self supervised task, which reinforces node representation learning via self-
    discrimination.Specifically,SGL generates multiple views of a node, maximizing the
    agreement between different views of the same node compared to that of other nodes.
    SGL devises three operators to generate the views — node dropout, edge dropout, and
    random walk — that change the graph structure in different manners.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SGL, self).__init__(config, dataset)
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]
        self.embed_dim = config["embedding_size"]
        self.n_layers = int(config["n_layers"])
        self.type = config["type"]
        self.drop_ratio = config["drop_ratio"]
        self.ssl_tau = config["ssl_tau"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]
        self.user_embedding = torch.nn.Embedding(self.n_users, self.embed_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.embed_dim)
        self.reg_loss = EmbLoss()
        self.train_graph = self.csr2tensor(self.create_adjust_matrix(is_sub=False))
        self.restore_user_e = None
        self.restore_item_e = None
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        conv_fn = partial(JacobiConv, a=config["a"], b=config["b"])
        # conv_fn_low = partial(PowerConv)
        # conv_fn_high = partial(PowerConv)
        # conv_fn = partial(ChebyshevConv)

        # self.low_list = config['low_list']
        # self.high_list = config['high_list']
        # self.low_weight = config['low_weight']
        # self.high_weight = config['high_weight']

        # self.step, self.filter_layers = config['step'], config['filter_layers']
        # self.k_low,self.i_low = config['k_low'],config['i_low']
        # self.k_high,self.i_high = config['k_high'],config['i_high']
        # self.filter_loss_weight = config['filter_loss_weight']

        self.graph_conv_low = PolyConvFrame(conv_fn=conv_fn, depth=self.n_layers, alpha=3.0)
        # # self.graph_conv_low = PolyConvFrame(conv_fn=conv_fn_low, depth=self.filter_layers, alpha=3.0)
        # # self.graph_conv_high = PolyConvFrame(conv_fn=conv_fn_high, depth=self.filter_layers, alpha=3.0)
        # self.graph_conv_low = PolyConvFrame(conv_fn=conv_fn_low, depth=self.filter_layers, alpha=3.0,alphas=config["low_alphas"])
        # self.graph_conv_high = PolyConvFrame(conv_fn=conv_fn_high, depth=self.filter_layers, alpha=3.0,alphas=config["high_alphas"])

        # self.train_graph_high,self.train_graph_low = self.get_high_low_filter_graph(self.train_graph,self.k_high,self.i_high,self.k_low,self.i_low)

        self.beta = config["alpha"]
        # self.ssl_weight_low = config["ssl_weight_low"]
        # self.ssl_weight_mid = config["ssl_weight_mid"]
        # self.ssl_weight_high = config["ssl_weight_high"]

        # self.weight_low,self.weight_mid,self.weight_high = config['weight_low'],config['weight_mid'],config['weight_high']

    # def get_high_low_filter_graph(self,graph,k_high,i_high,k_low,i_low):
    #     indices,values = graph._indices(),graph._values()
    #     high_filter_graph_values = high_filter(values,k_high,i_high)
    #     low_filter_graph_values = low_filter(values,k_low,i_low)
    #
    #     high_filter_graph = torch.sparse.FloatTensor(indices,high_filter_graph_values,size=graph.shape)
    #     low_filter_graph = torch.sparse.FloatTensor(indices,low_filter_graph_values,size=graph.shape)
    #
    #     return high_filter_graph,low_filter_graph

    def graph_construction(self):
        r"""Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node."""
        self.sub_graph1 = []
        if self.type == "ND" or self.type == "ED":
            self.sub_graph1 = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
        elif self.type == "RW":
            for i in range(self.n_layers):
                _g = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
                self.sub_graph1.append(_g)

        # self.sub_graph1_high, self.sub_graph1_low = self.get_high_low_filter_graph(self.sub_graph1, self.k_high,self.i_high, self.k_low,self.i_low)

        self.sub_graph2 = []
        if self.type == "ND" or self.type == "ED":
            self.sub_graph2 = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
        elif self.type == "RW":
            for i in range(self.n_layers):
                _g = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
                self.sub_graph2.append(_g)

        # self.sub_graph2_high, self.sub_graph2_low = self.get_high_low_filter_graph(self.sub_graph2, self.k_high,self.i_high, self.k_low, self.i_low)

    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly discard some points or edges.

        Args:
            high (int): Upper limit of index value
            size (int): Array size after sampling

        Returns:
            numpy.ndarray: Array index after sampling, shape: [size]
        """

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def create_adjust_matrix(self, is_sub: bool):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.If it is a subgraph, it may be processed by
        node dropout or edge dropout.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            csr_matrix of the normalized interaction matrix.
        """
        matrix = None
        if not is_sub:
            ratings = np.ones_like(self._user, dtype=np.float32)
            matrix = sp.csr_matrix(
                (ratings, (self._user, self._item + self.n_users)),
                shape=(self.n_users + self.n_items, self.n_users + self.n_items),
            )
        else:
            if self.type == "ND":
                drop_user = self.rand_sample(
                    self.n_users,
                    size=int(self.n_users * self.drop_ratio),
                    replace=False,
                )
                drop_item = self.rand_sample(
                    self.n_items,
                    size=int(self.n_items * self.drop_ratio),
                    replace=False,
                )
                R_user = np.ones(self.n_users, dtype=np.float32)
                R_user[drop_user] = 0.0
                R_item = np.ones(self.n_items, dtype=np.float32)
                R_item[drop_item] = 0.0
                R_user = sp.diags(R_user)
                R_item = sp.diags(R_item)
                R_G = sp.csr_matrix(
                    (
                        np.ones_like(self._user, dtype=np.float32),
                        (self._user, self._item),
                    ),
                    shape=(self.n_users, self.n_items),
                )
                res = R_user.dot(R_G)
                res = res.dot(R_item)

                user, item = res.nonzero()
                ratings = res.data
                matrix = sp.csr_matrix(
                    (ratings, (user, item + self.n_users)),
                    shape=(self.n_users + self.n_items, self.n_users + self.n_items),
                )

            elif self.type == "ED" or self.type == "RW":
                keep_item = self.rand_sample(
                    len(self._user),
                    size=int(len(self._user) * (1 - self.drop_ratio)),
                    replace=False,
                )
                user = self._user[keep_item]
                item = self._item[keep_item]

                matrix = sp.csr_matrix(
                    (np.ones_like(user), (user, item + self.n_users)),
                    shape=(self.n_users + self.n_items, self.n_users + self.n_items),
                )

        matrix = matrix + matrix.T
        D = np.array(matrix.sum(axis=1)) + 1e-7
        D = np.power(D, -0.5).flatten()
        D = sp.diags(D)
        return D.dot(matrix).dot(D)

    def csr2tensor(self, matrix: sp.csr_matrix):
        r"""Convert csr_matrix to tensor.

        Args:
            matrix (scipy.csr_matrix): Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([matrix.row, matrix.col])),
            torch.FloatTensor(matrix.data.astype(np.float32)),
            matrix.shape,
        ).to(self.device)
        # x = SparseTensor(row=torch.tensor(matrix.row, dtype=torch.long, device=self.device), col=torch.tensor(matrix.col, dtype=torch.long, device=self.device), value=torch.tensor(matrix.data.astype(np.float32), device=self.device), sparse_sizes=matrix.shape)
        return x

    def tensor2index(self,tensor):
        row,col,_ = tensor.coo()
        row,col = row.tolist(),col.tolist()
        edge_index = torch.tensor([row, col], dtype=torch.long, device=self.device)
        return edge_index

    def forward1(self, graph):
        main_ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_ego = [main_ego]
        if isinstance(graph, list):
            for sub_graph in graph:
                # sub_graph = from_torch_sparse(sub_graph.coalesce())
                main_ego = torch.sparse.mm(sub_graph, main_ego)
                all_ego.append(main_ego)
        else:
            # graph = from_torch_sparse(graph.coalesce())
            for i in range(self.n_layers):
                main_ego = torch.sparse.mm(graph, main_ego)
                all_ego.append(main_ego)
        all_ego = torch.stack(all_ego, dim=1)
        all_ego = torch.mean(all_ego, dim=1, keepdim=False)
        user_emd, item_emd = torch.split(all_ego, [self.n_users, self.n_items], dim=0)
        return user_emd, item_emd

    def forward(self,graph):

        all_embeddings = self.get_ego_embeddings()

        all_embeddings_low = self.graph_conv_low.forward1(all_embeddings, graph)

        all_embeddings_low = all_embeddings_low.mean(1)

        # all_embeddings_mid = self.beta * all_embeddings - all_embeddings_low

        # all_embeddings = torch.hstack([all_embeddings_low, all_embeddings_mid])

        all_embeddings = all_embeddings_low

        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    # def forward(self,graph):
    #     #JGCF
    #     all_embeddings = self.get_ego_embeddings()
    #     all_embeddings_low = self.graph_conv.forward1(all_embeddings, graph)
    #     all_embeddings_low = all_embeddings_low.mean(1)
    #     all_embeddings_mid = self.beta * all_embeddings - all_embeddings_low
    #     all_embeddings = torch.hstack([all_embeddings_low, all_embeddings_mid])
    #     # all_embeddings = all_embeddings_low
    #     user_all_embeddings, item_all_embeddings = torch.split(
    #         all_embeddings, [self.n_users, self.n_items]
    #     )
    #     return user_all_embeddings, item_all_embeddings#,0,0
    #
    # def forward2(self,graph):
    #     #SD-GCN
    #     all_embeddings = self.get_ego_embeddings()
    #     all_embeddings_low = self.graph_conv_low.forward1(all_embeddings, graph,a=self.low_list)
    #     all_embeddings_low = all_embeddings_low.mean(1)
    #     all_embeddings_high = self.graph_conv_high.forward1(all_embeddings,graph,a=self.high_list,high=True)
    #     all_embeddings_high = -all_embeddings_high.mean(1)
    #     all_embeddings = self.low_weight*all_embeddings_low+self.high_weight*all_embeddings_high
    #     user_all_embeddings, item_all_embeddings = torch.split(
    #         all_embeddings, [self.n_users, self.n_items]
    #     )
    #     return user_all_embeddings, item_all_embeddings#,all_embeddings_high,all_embeddings_low
    #
    # def forward3(self,graph):
    #     #high/low filter JGCF
    #     ori_all_embeddings = self.get_ego_embeddings()
    #     ori_all_embeddings_low = self.graph_conv.forward1(ori_all_embeddings, graph)
    #     ori_all_embeddings_low = ori_all_embeddings_low.mean(1)
    #     ori_all_embeddings_mid = self.beta * ori_all_embeddings - ori_all_embeddings_low
    #     all_embeddings = torch.hstack([ori_all_embeddings_low, ori_all_embeddings_mid])
    #     all_embeddings_high, all_embeddings_mid, all_embeddings_low = 0,0,0
    #     all_embeddings_low = self.graph_conv_low.forward1(all_embeddings, graph,filter=True)
    #     all_embeddings_low = all_embeddings_low.mean(1)
    #     all_embeddings_high = self.graph_conv_high.forward1(all_embeddings, graph,filter=True)
    #     all_embeddings_high = all_embeddings_high.mean(1)
    #     all_embeddings_mid = all_embeddings-all_embeddings_low-all_embeddings_high
    #     all_embeddings = self.weight_low*all_embeddings_low+self.weight_mid*all_embeddings_mid+self.weight_high*all_embeddings_high
    #     # all_embeddings = all_embeddings-all_embeddings_high
    #     user_all_embeddings, item_all_embeddings = torch.split(
    #         all_embeddings, [self.n_users, self.n_items]
    #     )
    #     return user_all_embeddings, item_all_embeddings#,all_embeddings_high,all_embeddings_mid,all_embeddings_low

    # def calculate_loss1(self, interaction):
    #     # JGCF
    #     if self.restore_user_e is not None or self.restore_item_e is not None:
    #         self.restore_user_e, self.restore_item_e = None, None
    #
    #     user_list = interaction[self.USER_ID]
    #     pos_item_list = interaction[self.ITEM_ID]
    #     neg_item_list = interaction[self.NEG_ITEM_ID]
    #     user_emd, item_emd = self.forward(self.train_graph)
    #     a = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)
    #     total_loss = a
    #     return total_loss

    def calculate_loss(self, interaction):
        #SGL
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        user_emd, item_emd = self.forward(self.train_graph)
        user_sub1, item_sub1 = self.forward(self.sub_graph1)
        user_sub2, item_sub2 = self.forward(self.sub_graph2)
        a = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)
        b = self.calc_ssl_loss(user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2,self.ssl_weight)
        total_loss = a + b
        return total_loss

    # def calculate_loss4(self, interaction):
    #     #spectral devide GCN(SD-GCN)
    #     if self.restore_user_e is not None or self.restore_item_e is not None:
    #         self.restore_user_e, self.restore_item_e = None, None
    #
    #     user_list = interaction[self.USER_ID]
    #     pos_item_list = interaction[self.ITEM_ID]
    #     neg_item_list = interaction[self.NEG_ITEM_ID]
    #     user_emd, item_emd = self.forward2(self.train_graph)
    #     a = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)
    #     total_loss = a
    #     return total_loss
    #
    # def calculate_loss3(self, interaction):
    #     #SD-GCN+SSL
    #     if self.restore_user_e is not None or self.restore_item_e is not None:
    #         self.restore_user_e, self.restore_item_e = None, None
    #
    #     user_list = interaction[self.USER_ID]
    #     pos_item_list = interaction[self.ITEM_ID]
    #     neg_item_list = interaction[self.NEG_ITEM_ID]
    #     user_emd, item_emd,_,_ = self.forward2(self.train_graph)
    #     _, _, emb1_high, emb1_low = self.forward2(self.sub_graph1)
    #     _, _, emb2_high, emb2_low = self.forward2(self.sub_graph2)
    #     user_high_sub1,item_high_sub1 = torch.split(emb1_high, [self.n_users, self.n_items])
    #     user_high_sub2, item_high_sub2 = torch.split(emb2_high, [self.n_users, self.n_items])
    #     user_low_sub1, item_low_sub1 = torch.split(emb1_low, [self.n_users, self.n_items])
    #     user_low_sub2, item_low_sub2 = torch.split(emb2_low, [self.n_users, self.n_items])
    #     a = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)
    #     b = self.calc_ssl_loss(user_list, pos_item_list, user_high_sub1, user_high_sub2, item_high_sub1, item_high_sub2,self.ssl_weight_high)
    #     c = self.calc_ssl_loss(user_list, pos_item_list, user_low_sub1, user_low_sub2, item_low_sub1, item_low_sub2,self.ssl_weight_low)
    #     total_loss = a + b + c
    #     return total_loss
    #
    # def calculate_loss5(self, interaction):
    #     #high/low filter JGCF + SSL
    #     if self.restore_user_e is not None or self.restore_item_e is not None:
    #         self.restore_user_e, self.restore_item_e = None, None
    #
    #     user_list = interaction[self.USER_ID]
    #     pos_item_list = interaction[self.ITEM_ID]
    #     neg_item_list = interaction[self.NEG_ITEM_ID]
    #     user_emd, item_emd,_,_,_ = self.forward3(self.train_graph)
    #     _, _, emb1_high,emb1_mid, emb1_low = self.forward3(self.sub_graph1)
    #     _, _, emb2_high,emb2_mid, emb2_low = self.forward3(self.sub_graph2)
    #     user_low_sub1, item_low_sub1 = torch.split(emb1_low, [self.n_users, self.n_items])
    #     user_low_sub2, item_low_sub2 = torch.split(emb2_low, [self.n_users, self.n_items])
    #     # user_mid_sub1, item_mid_sub1 = torch.split(emb1_mid, [self.n_users, self.n_items])
    #     # user_mid_sub2, item_mid_sub2 = torch.split(emb2_mid, [self.n_users, self.n_items])
    #     # user_high_sub1, item_high_sub1 = torch.split(emb1_high, [self.n_users, self.n_items])
    #     # user_high_sub2, item_high_sub2 = torch.split(emb2_high, [self.n_users, self.n_items])
    #     a = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)
    #     # b = self.calc_filter_loss()
    #     d = self.calc_ssl_loss(user_list, pos_item_list, user_low_sub1, user_low_sub2, item_low_sub1, item_low_sub2,self.ssl_weight_low)
    #     # e = self.calc_ssl_loss(user_list, pos_item_list, user_mid_sub1, user_mid_sub2, item_mid_sub1, item_mid_sub2,self.ssl_weight_mid)
    #     # c = self.calc_ssl_loss(user_list, pos_item_list, user_high_sub1, user_high_sub2, item_high_sub1, item_high_sub2,self.ssl_weight_high)
    #     total_loss = a + d#+e+c# + c
    #     return total_loss
    #
    # def calculate_loss(self, interaction):
    #     # high/low JGCF+SSL
    #     if self.restore_user_e is not None or self.restore_item_e is not None:
    #         self.restore_user_e, self.restore_item_e = None, None
    #
    #     user_list = interaction[self.USER_ID]
    #     pos_item_list = interaction[self.ITEM_ID]
    #     neg_item_list = interaction[self.NEG_ITEM_ID]
    #     user_emd, item_emd = self.forward3(self.train_graph)
    #     user_sub1, item_sub1 = self.forward3(self.sub_graph1)
    #     user_sub2, item_sub2 = self.forward3(self.sub_graph2)
    #     a = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)
    #     b = self.calc_ssl_loss(user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2, self.ssl_weight)
    #     total_loss = a + b
    #     return total_loss
    #
    # def calculate_loss6(self, interaction):
    #     #high/low filter JGCF
    #     if self.restore_user_e is not None or self.restore_item_e is not None:
    #         self.restore_user_e, self.restore_item_e = None, None
    #
    #     user_list = interaction[self.USER_ID]
    #     pos_item_list = interaction[self.ITEM_ID]
    #     neg_item_list = interaction[self.NEG_ITEM_ID]
    #     user_emd, item_emd = self.forward3(self.train_graph)
    #     a = self.calc_bpr_loss(user_emd, item_emd, user_list, pos_item_list, neg_item_list)
    #     # b = self.calc_filter_loss()
    #     total_loss = a #+ b
    #     return total_loss

    # def calc_filter_loss(self):
    #     x = torch.arange(-1,1,step=self.step).to(self.device)
    #
    #     low_filter_y = low_filter(x,self.k_low,self.i_low)
    #     pre_low_filter_y = fit_filter(x,self.graph_conv_low.alphas/(self.n_layers+1))
    #
    #     low_loss = torch.sqrt(torch.sum((low_filter_y-pre_low_filter_y)**2))
    #
    #     high_filter_y = high_filter(x,self.k_high,self.i_high)
    #     pre_high_filter_y = fit_filter(x,self.graph_conv_high.alphas/(self.n_layers+1))
    #
    #     high_loss = torch.sqrt(torch.sum((high_filter_y-pre_high_filter_y)**2))
    #
    #     filter_loss = (low_loss + high_loss)*self.filter_loss_weight
    #     return filter_loss

    def calc_bpr_loss(
        self, user_emd, item_emd, user_list, pos_item_list, neg_item_list
    ):
        r"""Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = user_emd[user_list]
        pi_e = item_emd[pos_item_list]
        ni_e = item_emd[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)

        l1 = torch.sum(-F.logsigmoid(p_scores - n_scores))

        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.reg_loss(u_e_p, pi_e_p, ni_e_p)

        return l1 + l2 * self.reg_weight

    def calc_ssl_loss(
        self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2,weight
    ):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)
        all_user2 = F.normalize(user_sub2, dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2, dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * weight

    # def calc_rec_loss(self,B, E,users,items):
    #     batch_user_E,batch_user_B = E[users],B[users]
    #     batch_item_E,batch_item_B = E[items+self.n_users],B[items+self.n_users]
    #
    #     batch_E = torch.cat([batch_user_E,batch_item_E],dim=0)
    #     batch_B = torch.cat([batch_user_B,batch_item_B],dim=0)
    #
    #     score1 = torch.sum(batch_E*batch_B, dim=1)
    #     loss1 = torch.sqrt(torch.sum((score1-1)**2))
    #     score2 = torch.mm(E,batch_B.T)
    #     loss2 = torch.sqrt(torch.sum((score2-0)**2))
    #
    #     score3 = torch.sum(E.T*B.T, dim=1)
    #     loss3 = torch.sqrt(torch.sum((score3-1)**2))
    #     score4 = torch.mm(B.T,E)
    #     loss4 = torch.sqrt(torch.sum((score4-0)**2))
    #
    #     # socre1 = torch.sum(E * B, dim=1)
    #     # loss1 = torch.sqrt(torch.sum((socre1 - 1) ** 2))
    #     # score2 = torch.mm(E, B.T)
    #     # loss2 = torch.sqrt(torch.sum((score2 - 0) ** 2))
    #     #
    #     # socre3 = torch.sum(E * B, dim=1)
    #     # loss3 = torch.sqrt(torch.sum((socre3 - 1) ** 2))
    #     # score4 = torch.mm(B.T, E)
    #     # loss4 = torch.sqrt(torch.sum((score4 - 0) ** 2))
    #     loss = loss1 + loss2 + loss3 + loss4
    #     return loss

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)
            # self.restore_user_e, self.restore_item_e = self.forward2(self.train_graph)
            # self.restore_user_e, self.restore_item_e = self.forward3(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)
            # self.restore_user_e, self.restore_item_e = self.forward2(self.train_graph)
            # self.restore_user_e, self.restore_item_e = self.forward3(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)

    def train(self, mode: bool = True):
        r"""Override train method of base class.The subgraph is reconstructed each time it is called."""
        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T
