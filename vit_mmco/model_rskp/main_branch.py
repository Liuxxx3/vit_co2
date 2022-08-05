import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.autograd import Variable
from modeling.backbones_vit import ConvBackbone,ConvTransformerBackbone_co2
from modeling.necks import FPNIdentity_co2
from modeling.meta_archs import PtTransformerClsHead_co2
from modeling.blocks import MaskedConv1D

def weights_init_random(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = f / (f_norm + 1e-9)
    return f

def random_walk(x, y, w):
    x_norm = calculate_l1_norm(x)
    y_norm = calculate_l1_norm(y)
    eye_x = torch.eye(x.size(1)).float().to(x.device)

    latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [y_norm, x_norm]) * 5.0, 1)
    norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
    affinity_mat = torch.einsum('nkt,nkd->ntd', [latent_z, norm_latent_z])
    # mat_inv_x, _ = torch.solve(eye_x, eye_x - (w ** 2) * affinity_mat)
    mat_inv_x = torch.inverse(eye_x - (w ** 2) * affinity_mat)
    y2x_sum_x = w * torch.einsum('nkt,nkd->ntd', [latent_z, y]) + x
    refined_x = (1 - w) * torch.einsum('ntk,nkd->ntd', [mat_inv_x, y2x_sum_x])    

    return refined_x

def weights_init(m):
        classname = m.__class__.__name__
        # print(classname)
        # import pdb
        # pdb.set_trace()
        names = ['ConvBackbone','MaskedConv1D','ConvBlock','ConvTransformerBackbone_co2']
        if classname not in names: 
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                # torch_init.xavier_uniform_(m.weight)
                # import pdb
                # pdb.set_trace()
                torch_init.kaiming_uniform_(m.weight)
                if type(m.bias)!=type(None):
                    m.bias.data.fill_(0)
class WSTAL(nn.Module):  
    def __init__(self,args):
        super().__init__()
        n_class = args.class_num
        n_feature = args.inp_feat_num
        embed_dim=2048
        mid_dim=1024
        fpn_dim = 1024
        dropout_ratio=0.7
        reduce_ratio=16
        self.max_seq_len = 2304
        self.n_mu = args.mu_num
        self.n_out = args.out_feat_num
        self.em_iter = args.em_iter
        self.w = args.w
        self.mu = nn.Parameter(torch.randn(self.n_mu, self.n_out))
        torch_init.xavier_uniform_(self.mu)
        self.device = torch.device(
            'cuda:' + str(args.gpu_ids[0]) if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu')
        self.fusion = nn.ModuleList()
        self.fusion.append(MaskedConv1D(in_channels = n_feature, out_channels = n_feature, 
            kernel_size = 1, stride=1,padding=0,bias=False))
        self.fusion.append(nn.LeakyReLU(0.2))
        self.fusion.append(nn.Dropout(dropout_ratio))
        self.backbone_v = ConvTransformerBackbone_co2(
                n_in = mid_dim,                  # input feature dimension
                n_embd = mid_dim,                 # embedding dimension (after convolution)
                n_head = 4,                # number of head for self-attention in transformers
                n_embd_ks = 3,             # conv kernel size of the embedding network
                max_len = 2304,               # max sequence length
                arch = (1, 1),      # (#convs, #stem transformers, #branch transformers)
                mha_win_size = [-1]*6, # size of local window for mha
                scale_factor = 2,      # dowsampling rate for the branch,
                with_ln = True,       # if to attach layernorm after conv
                attn_pdrop = 0.0,      # dropout rate for the attention map
                proj_pdrop = 0.0,      # dropout rate for the projection / MLP
                path_pdrop = 0.1,      # droput rate for drop path
                use_abs_pe = True,    # use absolute position embedding
                use_rel_pe = False,    # use relative position embedding
                )
        self.backbone_f = ConvTransformerBackbone_co2(
                n_in = mid_dim,                  # input feature dimension
                n_embd = mid_dim,                 # embedding dimension (after convolution)
                n_head = 4,                # number of head for self-attention in transformers
                n_embd_ks = 3,             # conv kernel size of the embedding network
                max_len = 2304,               # max sequence length
                arch = (1, 1),      # (#convs, #stem transformers, #branch transformers)
                mha_win_size = [-1]*6, # size of local window for mha
                scale_factor = 2,      # dowsampling rate for the branch,
                with_ln = True,       # if to attach layernorm after conv
                attn_pdrop = 0.0,      # dropout rate for the attention map
                proj_pdrop = 0.0,      # dropout rate for the projection / MLP
                path_pdrop = 0.1,      # droput rate for drop path
                use_abs_pe = True,    # use absolute position embedding
                use_rel_pe = False,    # use relative position embedding
                )
        self.attention_v = nn.ModuleList()        
        self.attention_v.append(MaskedConv1D(in_channels = 1024, out_channels = 512, kernel_size = 3, 
            stride=1,padding=1,bias=False))
        self.attention_v.append(nn.LeakyReLU(0.2))
        self.attention_v.append(nn.Dropout(0.5))
        self.attention_v.append(MaskedConv1D(in_channels = 512, out_channels = 512, kernel_size = 3, 
            stride=1,padding=1,bias=False))
        self.attention_v.append(nn.LeakyReLU(0.2)) 
        self.attention_v.append(MaskedConv1D(in_channels = 512, out_channels = 1, kernel_size = 1, 
            stride=1,padding=0,bias=False))
        self.attention_v.append(nn.Dropout(0.5))
        self.attention_v.append(nn.Sigmoid())

        self.attention_f = nn.ModuleList()
        self.attention_f.append(MaskedConv1D(in_channels = 1024, out_channels = 512, kernel_size = 3, 
            stride=1,padding=1,bias=False))
        self.attention_f.append(nn.LeakyReLU(0.2))
        self.attention_f.append(nn.Dropout(0.5))
        self.attention_f.append(MaskedConv1D(in_channels = 512, out_channels = 512, kernel_size = 3, 
            stride=1,padding=1,bias=False))
        self.attention_f.append(nn.LeakyReLU(0.2)) 
        self.attention_f.append(MaskedConv1D(in_channels = 512, out_channels = 1, kernel_size = 1, 
            stride=1,padding=0,bias=False))
        self.attention_f.append(nn.Dropout(0.5))
        self.attention_f.append(nn.Sigmoid())
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Dropout(dropout_ratio))
        self.classifier.append(MaskedConv1D(in_channels = embed_dim, out_channels = embed_dim, 
        kernel_size = 3, stride=1,padding=1,bias=True))
        self.classifier.append(nn.LeakyReLU(0.2))
        self.classifier.append(nn.Dropout(0.7))
        self.classifier.append(MaskedConv1D(in_channels = embed_dim, out_channels = n_class+1, 
        kernel_size = 1, stride=1,padding=0,bias=True))
        
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
        self.batch_avg=nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        
        self.apply(weights_init)

    def EM(self, mu, x):
        # propagation -> make mu as video-specific mu
        norm_x = calculate_l1_norm(x)
        for _ in range(self.em_iter):
            norm_mu = calculate_l1_norm(mu)
            latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0, 1)
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True)+1e-9)
            mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])
        return mu
    
    def feature_embedding(self,features,batched_masks_v):

        batched_masks_f = batched_masks_v
        feat_vs = features[:,:1024,:]
        feat_fs = features[:,1024:,:]
        feat_vs,mask_v = self.backbone_v(feat_vs,batched_masks_v)
        feat_fs,mask_f = self.backbone_f(feat_fs,batched_masks_f)
        return torch.cat((feat_vs,feat_fs),axis=1).transpose(-1,-2),mask_v,mask_f

    def PredictionModule(self, features,mask_v,mask_f):
        # pdb.set_trace()
        # print('1',features.device)
        if features.shape[1]!=2048:
            features = features.transpose(-1,-2)
        feat_vs = features[:,:1024,:]
        feat_fs = features[:,1024:,:]
        v_atn = feat_vs
        f_atn = feat_fs
        for idx in range(len(self.attention_v)):
            if 'MaskedConv1D' in str(self.attention_v[idx]):
                v_atn, mask_v = self.attention_v[idx](v_atn, mask_v)
            else:
                v_atn = self.attention_v[idx](v_atn)

        for idx in range(len(self.attention_f)):
            if 'MaskedConv1D' in str(self.attention_f[idx]):
                f_atn, mask_f = self.attention_f[idx](f_atn, mask_f)
            else:
                f_atn = self.attention_f[idx](f_atn)
        x_atn = ((f_atn+v_atn)/2)
        nfeat = torch.cat((feat_vs,feat_fs),1)
        # nfeat = self.fusion(nfeat)
        for idx in range(len(self.fusion)):
            if 'MaskedConv1D' in str(self.fusion[idx]):
                nfeat, mask_f = self.fusion[idx](nfeat, mask_f)
            else:
                nfeat = self.fusion[idx](nfeat)
        
        # x_cls = self.classifier(nfeat)
        x_cls = nfeat
        for idx in range(len(self.classifier)):
            if 'MaskedConv1D' in str(self.classifier[idx]):
                x_cls, mask_f = self.classifier[idx](x_cls, mask_f)
            else:
                x_cls = self.classifier[idx](x_cls)
        return {'feat':nfeat.transpose(-1, -2), 'cas':x_cls.transpose(-1, -2), 'attn':x_atn.transpose(-1, -2),\
            'v_atn':v_atn.transpose(-1, -2),'f_atn':f_atn.transpose(-1, -2),'mask':mask_f.transpose(-1, -2)}

    def forward(self, x,mask):
        n, d,t = x.size()

        # feature embedding
        x,mask_v,mask_f = self.feature_embedding(x,mask)

        # Expectation Maximization of class agnostic tokens
        mu = self.mu[None, ...].repeat(n, 1, 1)
        mu = self.EM(mu, x)
        # feature reallocate
        reallocated_x = random_walk(x, mu, self.w)

        # original feature branch
        outputs_origin = self.PredictionModule(x,mask_v,mask_f)
        # reallocated feature branch
        outputs_reallocated = self.PredictionModule(reallocated_x,mask_v,mask_f)

        # mu classification scores
        norms_mu = calculate_l1_norm(mu)
        # norms_ac = calculate_l1_norm(self.ac_center)
        # mu_scr = torch.einsum('nkd,cd->nkc', [norms_mu, norms_ac]) * self.scale_factor
        # mu_pred = F.softmax(mu_scr, -1)
        x_cls = norms_mu.transpose(-1,-2)
        mask_mu = torch.arange(self.n_mu)[None, :] < torch.full([1],self.max_seq_len)[:, None]
        mask_mu = mask_mu.to(self.device)

        for idx in range(len(self.classifier)):
            if 'MaskedConv1D' in str(self.classifier[idx]):
                x_cls, mask_mu = self.classifier[idx](x_cls, mask_mu)
            else:
                x_cls = self.classifier[idx](x_cls)
        mu_pred = x_cls.transpose(-1,-2)
        return outputs_origin,\
               outputs_reallocated,\
               [x, mu, mu_pred]

# class WSTAL(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         # feature embedding
#         self.w = args.w
#         self.n_in = args.inp_feat_num
#         self.n_out = args.out_feat_num

#         self.n_mu = args.mu_num
#         self.em_iter = args.em_iter
#         self.n_class = args.class_num
#         self.scale_factor = args.scale_factor
#         self.dropout = args.dropout

#         self.mu = nn.Parameter(torch.randn(self.n_mu, self.n_out))
#         torch_init.xavier_uniform_(self.mu)

#         self.ac_center = nn.Parameter(torch.randn(self.n_class + 1, self.n_out))
#         torch_init.xavier_uniform_(self.ac_center)
#         self.fg_center = nn.Parameter(-1.0 * self.ac_center[-1, ...][None, ...])

#         self.feature_embedding = nn.Sequential(
#                                     nn.Linear(self.n_in, self.n_out),
#                                     nn.ReLU(inplace=True),
#                                     nn.Dropout(self.dropout),
#                                     )

#         self.sigmoid = nn.Sigmoid()
#         self.apply(weights_init_random)

#     def EM(self, mu, x):
#         # propagation -> make mu as video-specific mu
#         norm_x = calculate_l1_norm(x)
#         for _ in range(self.em_iter):
#             norm_mu = calculate_l1_norm(mu)
#             latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0, 1)
#             norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True)+1e-9)
#             mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])
#         return mu

#     def PredictionModule(self, x):
#         # normalization
#         norms_x = calculate_l1_norm(x)
#         norms_ac = calculate_l1_norm(self.ac_center)
#         norms_fg = calculate_l1_norm(self.fg_center)

#         # generate class scores        
#         frm_fb_scrs = torch.einsum('ntd,kd->ntk', [norms_x, norms_fg]).squeeze(-1) * self.scale_factor#k==1,bg层        
#         frm_scrs = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * self.scale_factor

#         # generate attention
#         class_agno_att = self.sigmoid(frm_fb_scrs)
#         class_wise_att = self.sigmoid(frm_scrs)
#         class_agno_norm_att = class_agno_att / (torch.sum(class_agno_att, dim=1, keepdim=True) + 1e-5)
#         class_wise_norm_att = class_wise_att / (torch.sum(class_wise_att, dim=1, keepdim=True) + 1e-5)

#         ca_vid_feat = torch.einsum('ntd,nt->nd', [x, class_agno_norm_att])
#         cw_vid_feat = torch.einsum('ntd,ntc->ncd', [x, class_wise_norm_att])

#         # normalization
#         norms_ca_vid_feat = calculate_l1_norm(ca_vid_feat)
#         norms_cw_vid_feat = calculate_l1_norm(cw_vid_feat)

#         # classification
#         frm_scr = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * self.scale_factor
#         ca_vid_scr = torch.einsum('nd,cd->nc', [norms_ca_vid_feat, norms_ac]) * self.scale_factor
#         cw_vid_scr = torch.einsum('ncd,cd->nc', [norms_cw_vid_feat, norms_ac]) * self.scale_factor

#         # prediction
#         ca_vid_pred = F.softmax(ca_vid_scr, -1)
#         cw_vid_pred = F.softmax(cw_vid_scr, -1)

#         return ca_vid_pred, cw_vid_pred, class_agno_att, frm_scr

#     def forward(self, x):
#         n, t, _ = x.size()

#         # feature embedding
#         x = self.feature_embedding(x)

#         # Expectation Maximization of class agnostic tokens
#         mu = self.mu[None, ...].repeat(n, 1, 1)
#         mu = self.EM(mu, x)
#         # feature reallocate
#         reallocated_x = random_walk(x, mu, self.w)

#         # original feature branch
#         o_vid_ca_pred, o_vid_cw_pred, o_att, o_frm_pred = self.PredictionModule(x)
#         # reallocated feature branch
#         m_vid_ca_pred, m_vid_cw_pred, m_att, m_frm_pred = self.PredictionModule(reallocated_x)

#         # mu classification scores
#         norms_mu = calculate_l1_norm(mu)
#         norms_ac = calculate_l1_norm(self.ac_center)
#         mu_scr = torch.einsum('nkd,cd->nkc', [norms_mu, norms_ac]) * self.scale_factor
#         mu_pred = F.softmax(mu_scr, -1)

#         return [o_vid_ca_pred, o_vid_cw_pred, o_att, o_frm_pred],\
#                [m_vid_ca_pred, m_vid_cw_pred, m_att, m_frm_pred],\
#                [x, mu, mu_pred]