import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
from model_GCN import GCNII_lyc
import ipdb
from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from high_fre_conv import highConv

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from SeqContext import SeqContext


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp + i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx, a], dim=0)
            tmp = tmp + i
        # x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * feature_dim), # IEMOCAP
            # nn.Linear(2 * input_dim, 2 * feature_dim), # MELD
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # 第一层：线性层，将输入特征映射到隐藏层维度
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 第二层：线性层，将隐藏层映射到输出维度
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 激活函数
        self.relu = nn.ReLU()
        # Dropout (如果需要)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 输入到第一层
        x = self.fc1(x)
        # 使用ReLU激活函数
        x = self.relu(x)
        # Dropout层（用于正则化，防止过拟合）
        x = self.dropout(x)
        # 输入到第二层
        x = self.fc2(x)
        return x

class HyperGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant,
                 return_feature, use_residue, device,
                 M=0.1, n_speakers=2, modals=['a', 'v', 'l'], use_speaker=True, use_modal=False, num_L=3, num_K=4, dataset='IEMOCAP'):
        super(HyperGCN, self).__init__()

        self.device = device
        self.return_feature = return_feature  # True
        self.use_residue = use_residue
        self.new_graph = 'full'
        self.dataset = dataset  # 保存数据集名称

        # 根据数据集调整维度
        if dataset == 'MELD':
            # MELD的维度配置
            self.feature_dim = 1024  # 可能需要调整
            self.fc1 = nn.Linear(192, nhidden)
            self.fc2 = nn.Linear(n_dim, nhidden)
        else:
            # IEMOCAP的维度配置
            self.feature_dim = 1024
            self.fc1 = nn.Linear(192, nhidden)
            self.fc2 = nn.Linear(n_dim, nhidden)

        # self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
        #                       dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
        #                       return_feature=return_feature, use_residue=use_residue)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        # ------------------------------------
        self.fc1 = nn.Linear(192, nhidden)
        self.fc2 = nn.Linear(n_dim, nhidden)
        self.feature_reduce = nn.Linear(1024, 512)
        self.num_L = num_L
        self.num_K = num_K
        for ll in range(num_L):
            setattr(self, 'hyperconv%d' % (ll + 1), HypergraphConv(nhidden, nhidden))
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))
        # nn.init.xavier_uniform_(self.hyperedge_attr1)
        for kk in range(num_K):
            setattr(self, 'conv%d' % (kk + 1), highConv(nhidden, nhidden))

        # 添加共识情感学习单元的属性
        self.num_view = 3
        self.view_list = [512, 512, 512]
        self.feature_dim = 1024
        self.token_num = 16

        self.encoders = []  # dim encoder
        for v in range(self.num_view):
            self.encoders.append(Encoder(self.view_list[v], self.feature_dim).to(self.device))
        self.encoders = nn.ModuleList(self.encoders)
        # 记忆矩阵初始化
        self.consensus_prompts = nn.Parameter(torch.randn(self.num_view, self.token_num, self.feature_dim))

        self.aware_layers = []
        for v in range(self.num_view):
            '''源码'''
            self.aware_layer = SeqContext(self.feature_dim, self.feature_dim // 2, self.device)
            # self.aware_layer = SeqContext(self.feature_dim, 512, self.device)
            self.aware_layers.append(self.aware_layer.to(self.device))
        self.aware_layers = nn.ModuleList(self.aware_layers)



    def forward(self, a, v, l, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x, i, :] for i, x in enumerate(dia_len)], dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)#[869, 512]
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        if self.use_modal:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        '''共识感知'''
        data = {
            'input_tensor': torch.cat([a, v, l], dim=-1),
            'text_len_tensor': dia_len
        }
        data_unimodal, aware_features = self.get_aware(data, a, v, l)
        # aware_features = torch.cat(aware_features, dim=-1)
        # aware_features = self.fc1(aware_features) #[869, 512]

        # output_dim = 256  # 我们希望将两者映射到相同的维度 256
        # # 实例化两个MLP层分别处理文本和图像特征
        # mlp_text = MLP(input_dim=aware_features, hidden_dim=128, output_dim=output_dim)

        '''---------------------------------------Hyper GCN-----------------------------------------------'''
        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(a, v, l, dia_len, self.modals)
        x1 = self.fc2(features)
        weight = self.hyperedge_weight[0:hyperedge_index[1].max().item() + 1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        edge_attr = self.hyperedge_attr1 * hyperedge_type1 + self.hyperedge_attr2 * (1 - hyperedge_type1)
        out = x1 # [2607,512]
        '''结合共识感知的特征进行卷积'''
        out_new = torch.cat([out, aware_features], dim=0) # torch.Size([5214, 512])
        for ll in range(self.num_L):
            out = getattr(self, 'hyperconv%d' % (ll + 1))(out_new, hyperedge_index, weight, edge_attr, EW_weight, dia_len) # [2607, 512]
        out = torch.cat([out, out], dim=0)
        '''只使用共识感知特征进行卷积'''
        # out_hyper_aware = aware_features
        # for ll in range(self.num_L):
        #     out_hyper_aware = getattr(self, 'hyperconv%d' % (ll + 1))(out_hyper_aware, hyperedge_index, weight, edge_attr, EW_weight, dia_len)

        # if self.use_residue:
        #     out1 = torch.cat([features, out], dim=-1)
            # out2 = torch.cat([features, out2], dim=-1)
        # out1 = self.reverse_features(dia_len, out1)

        '''----------------------------------------High GCN---------------------------------------'''
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        gnn_out = x1

        '''结合共识感知的特征进行卷积'''
        gnn_out = torch.cat([gnn_out, aware_features], dim=0)
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self, 'conv%d' % (kk + 1))(gnn_out, gnn_edge_index) # [2607, 512]
        gnn_out = torch.cat([gnn_out, gnn_out], dim=0)  # [5214, 512]
        '''只使用共识感知特征进行卷积'''
        # out_high_aware = aware_features
        # for kk in range(self.num_K):
        #     out_high_aware = gnn_out + getattr(self, 'conv%d' % (kk + 1))(out_high_aware, gnn_edge_index)

        '''源码'''
        # out2 = torch.cat([out, gnn_out], dim=1)
        '''直接拼接aware_features'''
        # out2 = torch.cat([out, gnn_out, aware_features], dim=1)
        '''将aware_features进行卷积后拼接'''
        # out2 = torch.cat([out, gnn_out, out_hyper_aware], dim=1)
        # out2 = torch.cat([out, gnn_out, out_high_aware], dim=0)
        # out2 = torch.cat([out, gnn_out, out_high_aware, out_hyper_aware], dim=1)
        '''将aware_features与Hyper features拼接后，进行Hyper卷积'''
        # out2 = torch.cat([out, gnn_out], dim=0)
        '''将aware_features与High features拼接后，进行High卷积'''
        # out2 = torch.cat([out, gnn_out], dim=0)
        '''将aware_features与Hyper features,High features拼接后，进行Hyper,High卷积'''
        out2 = torch.cat([out, gnn_out], dim=1)


        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)

        # 保存最终特征（在reverse_features之前）
        final_features_before_reverse = out2.clone()  # 这是最终的特征表示
        out1 = self.reverse_features(dia_len, out2)
        return out1, aware_features, spk_emb_vector, data, final_features_before_reverse

    def get_aware(self, data, data_a, data_v, data_t):
        data_unimodal = []
        data_unimodal.append(data_a)  # 对应论文h^{a}
        data_unimodal.append(data_t)  # 对应论文h^{t}
        data_unimodal.append(data_v)  # 对应论文h^{v}

        out_dims = []
        for v in range(self.num_view):
            out_dim = self.encoders[v](data_unimodal[v])  # 对应论文z^{a}、z^{t}、z^{v}
            out_dims.append(out_dim)

        out_feats = torch.cat(out_dims, dim=-1)
        data['input_tensor'] = out_feats
        batch_diag_len = data['text_len_tensor']

        # 特征的重组和裁剪
        # bsz, ndiag, feat_dim = out_dims[0].size()
        # out_diag_views = []
        # for v in range(self.num_view):
        #     tmps = []
        #     for b in range(bsz):
        #         tmp = out_dims[v][b, :batch_diag_len[b], :]
        #         tmps.append(tmp)
        #     out_diag_views.append(torch.cat(tmps, dim=0))
        #
        # aware_features = []
        # for v in range(self.num_view):
        #     out_diag_view = out_diag_views[v]
        #     out_diag_view = out_diag_view.unsqueeze(1)
        #     view_consensus_prompt = self.consensus_prompts[v, :, :]
        #     view_consensus_prompts = view_consensus_prompt.repeat(out_diag_view.size(0), 1, 1)
        #     view_input = torch.cat([view_consensus_prompts, out_diag_view], dim=1)
        #     tmp_output = self.aware_layers[v](data["text_len_tensor"], view_input)
        #     aware_features.append(tmp_output[:, -1, :])

        aware_features = []
        for v in range(self.num_view):
            out_diag_view = out_dims[v]
            out_diag_view = out_diag_view.unsqueeze(1)
            view_consensus_prompt = self.consensus_prompts[v, :, :]
            view_consensus_prompts = view_consensus_prompt.repeat(out_diag_view.size(0), 1, 1)
            view_input = torch.cat([view_consensus_prompts, out_diag_view], dim=1)
            tmp_output = self.aware_layers[v](data["text_len_tensor"], view_input)
            aware_features.append(tmp_output[:, -1, :])
            aware_features_new = torch.cat(aware_features, dim=0)

        return data_unimodal, aware_features_new

    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 = []
        index2 = []
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []
        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]
            index1 = index1 + nodes_l + nodes_a + nodes_v
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]]
            for _ in range(i + 3):
                if _ < 3:
                    index2 = index2 + [edge_count] * i
                else:
                    index2 = index2 + [edge_count] * 3
                edge_count = edge_count + 1
            if node_count == 0:
                ll = l[0:0 + i]
                aa = a[0:0 + i]
                vv = v[0:0 + i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = 0 + i
            else:
                ll = l[temp:temp + i]
                aa = a[temp:temp + i]
                vv = v[temp:temp + i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features = torch.cat([features, features_temp], dim=0)
                temp = temp + i

            Gnodes = []
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_, 2))
                tmp = tmp + perm
            batch = batch + [batch_count] * i * 3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1] * i + [0] * 3

            node_count = node_count + i * num_modality

        index1 = torch.LongTensor(index1).view(1, -1)
        index2 = torch.LongTensor(index2).view(1, -1)
        hyperedge_index = torch.cat([index1, index2], dim=0).cuda()
        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([torch.arange(0, max_node + 1, 1).repeat_interleave(2).view(1, -1),
                               torch.arange(max_edge + 1, max_edge + 1 + max_node + 1, 1).repeat_interleave(2).view(1,
                                                                                                                    -1)],
                              dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()

        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1, 1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1

    def reverse_features(self, dia_len, features):
        l = []
        a = []
        v = []
        for i in dia_len:
            ll = features[0:1 * i]
            aa = features[1 * i:2 * i]
            vv = features[2 * i:3 * i]
            features = features[3 * i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l, dim=0)
        tmpa = torch.cat(a, dim=0)
        tmpv = torch.cat(v, dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features

    def create_gnn_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        batch_count = 0
        index = []
        tmp = []

        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]
            index = index + list(permutations(nodes_l, 2)) + list(permutations(nodes_a, 2)) + list(
                permutations(nodes_v, 2))
            Gnodes = []
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp + list(permutations(_, 2))
            if node_count == 0:
                ll = l[0:0 + i]
                aa = a[0:0 + i]
                vv = v[0:0 + i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = 0 + i
            else:
                ll = l[temp:temp + i]
                aa = a[temp:temp + i]
                vv = v[temp:temp + i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features = torch.cat([features, features_temp], dim=0)
                temp = temp + i
            node_count = node_count + i * num_modality
        edge_index = torch.cat([torch.LongTensor(index).T, torch.LongTensor(tmp).T], 1).cuda()

        return edge_index, features


def SupConLoss(temperature=1., contrast_mode='all', features=None, labels=None, mask=None, weights=None):
    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    # print(batch_size)
    # print(labels.shape[0])
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  # 1 indicates two items belong to same class
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # num of views
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (bsz * views, dim)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature  # (bsz * views, dim)
        anchor_count = contrast_count  # num of views
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    '''compute logits'''
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature)  # (bsz, bsz)
    '''for numerical stability'''
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (bsz, 1)
    logits = anchor_dot_contrast - logits_max.detach()  # (bsz, bsz) set max_value in logits to zero

    '''tile mask'''
    mask = mask.repeat(anchor_count, contrast_count)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                                0)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    mask = mask * logits_mask  # 1 indicates two items belong to same class and mask-out itself
    if weights is not None:
        mask = torch.mul(mask, weights)

    '''compute log_prob'''
    exp_logits = torch.exp(logits) * logits_mask  # (anchor_cnt * bsz, contrast_cnt * bsz)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    '''compute mean of log-likelihood over positive'''
    if 0 in mask.sum(1):
        raise ValueError('Make sure there are at least two instances with the same class')
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    # loss
    # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss

# 情感共识原型学习（类内损失和类间损失）
class My_anchor_Loss(nn.Module):
    def __init__(self, num_view, num_classes, feature_dim, Margin, size_average=True):
        super(My_anchor_Loss, self).__init__()
        self.num_view = num_view
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.size_average = size_average
        self.M = Margin
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.anchor = nn.Parameter(torch.randn(self.num_view, self.num_classes, self.feature_dim // 2).to(device))

    def forward(self, feat, label):

        center_losses = []# 类内损失
        inter_class_losses = []# 类间损失
        sample_similarities = []

        batch_size = feat[0].size(0)
        batch_size_tensor = feat[0].new_empty(1).fill_(
            batch_size if self.size_average else 1)

        for v in range(self.num_view):
            z = feat[v]
            centers_batch = self.anchor[v].index_select(0, label.long())
            sample_similarity = F.cosine_similarity(z, centers_batch)
            ####### intra_loss between center and sample
            center_loss = (z - centers_batch).pow(2).sum() / 2.0 / batch_size_tensor
            # ----------------------------------
            # print(center_loss)
            # ----------------------------------
            center_losses.append(center_loss)
            sample_similarities.append(sample_similarity)

        for v in range(self.num_view):
            centers_batch = self.anchor[v]

            # ######## inter_loss of centers
            inter_class_loss = torch.cuda.FloatTensor([0.])
            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    inter_class_loss += (centers_batch[i] - centers_batch[j]).pow(2).sum() / self.num_classes / (
                                self.num_classes - 1)
            # --------------------------------------
            # print(inter_class_loss)
            # --------------------------------------
            inter_class_losses.append(max(self.M - inter_class_loss, torch.cuda.FloatTensor([0.])))

        loss = sum(center_losses) / self.num_view + \
               sum(inter_class_losses) / self.num_view

        return loss, sample_similarities







