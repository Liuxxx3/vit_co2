from pickle import FALSE
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import model
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool
# from vit_cls import VisionTransformer as VisionTransformer_cls
# from vit_atn import VisionTransformer as VisionTransformer_atn
from vit import VisionTransformer
from modeling.backbones_vit import ConvBackbone,ConvTransformerBackbone_co2
from modeling.necks import FPNIdentity_co2
from modeling.meta_archs import PtTransformerClsHead_co2
from modeling.blocks import MaskedConv1D
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

class BWA_fusion_dropout_feat_v2(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        # self.attention = VisionTransformer(in_chans=1024,num_classes=1)
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
    def forward(self,vfeat,ffeat):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn*channel_attn)*vfeat#M:filter_feat;bit_wise_attn:Mg；channel_attn：Ml；vfeat：Xrgb
        x_atn = self.attention(filter_feat)
        return x_atn,filter_feat

class SE(torch.nn.Module):
    def __init__(self, n_feature,ratio,**args):        
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.compress = nn.Sequential(
            nn.Conv1d(n_feature, n_feature//ratio, kernel_size=1),
            nn.LeakyReLU(0.2))
        self.excitation = nn.Sequential(
            nn.Conv1d(n_feature//ratio, n_feature, kernel_size=1),
            nn.Sigmoid())
    def forward(self,feat):
        #(10,2048,320)
        a_feat = self.squeeze(feat)
        a_feat = self.compress(a_feat)
        a_feat = self.excitation(a_feat)
        filter_feat = feat*a_feat
        return filter_feat

class Inception(torch.nn.Module):
    def __init__(self, n_feature,**args):        
        super().__init__()
        dropout_ratio = 0.3
        self.vconv1 = nn.Sequential(
            nn.Conv1d(n_feature,n_feature,kernel_size=3,padding=1),
            nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio)
            )
        self.vconv2 = nn.Sequential(
            nn.Conv1d(n_feature,n_feature,kernel_size=5,padding=2),
            nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio)
            )
        self.fconv1 = nn.Sequential(
            nn.Conv1d(n_feature,n_feature,kernel_size=3,padding=1),
            nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio)
            )
        self.fconv2 = nn.Sequential(
            nn.Conv1d(n_feature,n_feature,kernel_size=5,padding=2),
            nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio)
            )
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature*4,n_feature*2,kernel_size=3,padding=1),
            nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio)
            )
    def forward(self,feat):
        vfeat = feat[:,:1024,:]
        ffeat = feat[:,1024:,:]
        vfeat_x = self.vconv1(vfeat)
        vfeat_y = self.vconv2(vfeat)
        ffeat_x = self.vconv1(ffeat)
        ffeat_y = self.vconv2(ffeat)
        output = torch.cat([vfeat_x,vfeat_y,ffeat_x,ffeat_y],dim = 1)
        output = self.fusion(output)
        return output


#fusion split modal single+ bit_wise_atten dropout+ contrastive + mutual learning +fusion feat(cat)
#------TOP!!!!!!!!!!
class CO2(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim=2048
        mid_dim=1024
        fpn_dim = 1024
        dropout_ratio=args['opt'].dropout_ratio
        reduce_ratio=args['opt'].reduce_ratio
        self.max_seq_len = 2304
        # self.vAttn = getattr(model,args['opt'].AWM)(1024,args)
        # self.fAttn = getattr(model,args['opt'].AWM)(1024,args)

        # self.feat_encoder = nn.Sequential(
        #     nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        # self.fusion = nn.Sequential(
        #     nn.Conv1d(n_feature, n_feature, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.fusion = nn.ModuleList()
        self.fusion.append(MaskedConv1D(in_channels = n_feature, out_channels = n_feature, 
            kernel_size = 1, stride=1,padding=0,bias=True))
        self.fusion.append(nn.LeakyReLU(0.2))
        self.fusion.append(nn.Dropout(dropout_ratio))
        # self.fusion = 
        # self.backbone_v = ConvBackbone(
        #             n_in=mid_dim,               # input feature dimension
        #             n_embd=mid_dim,             # embedding dimension (after convolution)
        #             n_embd_ks=3,          # conv kernel size of the embedding network
        #             arch = (2, 2, 5),      # (#convs, #stem convs, #branch convs)
        #             scale_factor = 2,    # dowsampling rate for the branch
        #             with_ln=True
        #         )
        # self.backbone_f = ConvBackbone(
        #             n_in=mid_dim,               # input feature dimension
        #             n_embd=mid_dim,             # embedding dimension (after convolution)
        #             n_embd_ks=3,          # conv kernel size of the embedding network
        #             arch = (2, 2, 5),      # (#convs, #stem convs, #branch convs)
        #             scale_factor = 2,    # dowsampling rate for the branch
        #             with_ln=True
        #         )
        self.backbone_v = ConvTransformerBackbone_co2(
                n_in = mid_dim,                  # input feature dimension
                n_embd = mid_dim,                 # embedding dimension (after convolution)
                n_head = 4,                # number of head for self-attention in transformers
                n_embd_ks = 3,             # conv kernel size of the embedding network
                max_len = 320,               # max sequence length
                arch = (1, 1),      # (#convs, #stem transformers, #branch transformers)
                mha_win_size = [-1]*6, # size of local window for mha
                scale_factor = 2,      # dowsampling rate for the branch,
                with_ln = True,       # if to attach layernorm after conv
                attn_pdrop = 0.0,      # dropout rate for the attention map
                proj_pdrop = 0.0,      # dropout rate for the projection / MLP
                path_pdrop = 0.1,      # droput rate for drop path
                use_abs_pe = False,    # use absolute position embedding
                use_rel_pe = False,    # use relative position embedding
                )
        self.backbone_f = ConvTransformerBackbone_co2(
                n_in = mid_dim,                  # input feature dimension
                n_embd = mid_dim,                 # embedding dimension (after convolution)
                n_head = 4,                # number of head for self-attention in transformers
                n_embd_ks = 3,             # conv kernel size of the embedding network
                max_len = 320,               # max sequence length
                arch = (1, 1),      # (#convs, #stem transformers, #branch transformers)
                mha_win_size = [-1]*6, # size of local window for mha
                scale_factor = 2,      # dowsampling rate for the branch,
                with_ln = True,       # if to attach layernorm after conv
                attn_pdrop = 0.0,      # dropout rate for the attention map
                proj_pdrop = 0.0,      # dropout rate for the projection / MLP
                path_pdrop = 0.1,      # droput rate for drop path
                use_abs_pe = False,    # use absolute position embedding
                use_rel_pe = False,    # use relative position embedding
                )
        # self.neck_v  = FPNIdentity_co2(
        #     in_channels =  mid_dim,
        #     out_channel =  mid_dim,
        #     scale_factor =  2,
        #     with_ln = True
        # )
        # self.neck_f  = FPNIdentity_co2(
        #     in_channels =  mid_dim,
        #     out_channel =  mid_dim,
        #     scale_factor =  2,
        #     with_ln = True
        # )
        # self.cls_head = PtTransformerClsHead_co2(
        #     input_dim = embed_dim,
        #     feat_dim = mid_dim, 
        #     num_classes = n_class+1,
        #     kernel_size=3,
        #     prior_prob=0.01,
        #     with_ln=True,
        #     num_layers=1,
        #     empty_cls=[]
        # )
        # self.attention_v = nn.Sequential(nn.Conv1d(mid_dim, 512, 3, padding=1),
        #                                nn.LeakyReLU(0.2),
        #                                nn.Dropout(0.5),
        #                                nn.Conv1d(512, 512, 3, padding=1),
        #                                nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
        #                                nn.Dropout(0.5),
        #                                nn.Sigmoid())
        # self.attention_f = nn.Sequential(nn.Conv1d(mid_dim, 512, 3, padding=1),
        #                                nn.LeakyReLU(0.2),
        #                                nn.Dropout(0.5),
        #                                nn.Conv1d(512, 512, 3, padding=1),
        #                                nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
        #                                nn.Dropout(0.5),
        #                                nn.Sigmoid())
        self.attention_v = nn.ModuleList()        
        self.attention_v.append(MaskedConv1D(in_channels = 1024, out_channels = 512, kernel_size = 3, 
            stride=1,padding=1,bias=True))
        self.attention_v.append(nn.LeakyReLU(0.2))
        self.attention_v.append(nn.Dropout(0.5))
        self.attention_v.append(MaskedConv1D(in_channels = 512, out_channels = 512, kernel_size = 3, 
            stride=1,padding=1,bias=True))
        self.attention_v.append(nn.LeakyReLU(0.2)) 
        self.attention_v.append(MaskedConv1D(in_channels = 512, out_channels = 1, kernel_size = 1, 
            stride=1,padding=0,bias=True))
        self.attention_v.append(nn.Dropout(0.5))
        self.attention_v.append(nn.Sigmoid())

        self.attention_f = nn.ModuleList()
        self.attention_f.append(MaskedConv1D(in_channels = 1024, out_channels = 512, kernel_size = 3, 
            stride=1,padding=1,bias=True))
        self.attention_f.append(nn.LeakyReLU(0.2))
        self.attention_f.append(nn.Dropout(0.5))
        self.attention_f.append(MaskedConv1D(in_channels = 512, out_channels = 512, kernel_size = 3, 
            stride=1,padding=1,bias=True))
        self.attention_f.append(nn.LeakyReLU(0.2)) 
        self.attention_f.append(MaskedConv1D(in_channels = 512, out_channels = 1, kernel_size = 1, 
            stride=1,padding=0,bias=True))
        self.attention_f.append(nn.Dropout(0.5))
        self.attention_f.append(nn.Sigmoid())
        # self.atn_head_v = PtTransformerClsHead_co2(
        #     input_dim = mid_dim,
        #     feat_dim = fpn_dim, 
        #     num_classes = 1,
        #     kernel_size=3,
        #     prior_prob=0.01,
        #     with_ln=True,
        #     num_layers=3,
        #     empty_cls=[]
        # )
        # self.atn_head_f = PtTransformerClsHead_co2(
        #     input_dim = mid_dim,
        #     feat_dim = fpn_dim, 
        #     num_classes = 1,
        #     kernel_size=3,
        #     prior_prob=0.01,
        #     with_ln=True,
        #     num_layers=3,
        #     empty_cls=[]
        # )
        # self.attention_v = VisionTransformer(t_length=320,in_chans=1024,num_classes=1,act_layer=nn.GELU(),\
        # mlp_ratio=1,depth=1,last_act=nn.Sigmoid())
        # self.attention_f = VisionTransformer(t_length=320,in_chans=1024,num_classes=1,act_layer=nn.GELU(),\
        # mlp_ratio=1,depth=1,last_act=nn.Sigmoid())
        # self.classifier = VisionTransformer(t_length=320,num_classes=n_class+1,mlp_ratio=1,act_layer=nn.GELU(),\
        # drop_rate=0.5,depth=1,last_act=None)
        
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout_ratio),
        #     nn.Conv1d(embed_dim, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),
        #     nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class+1, 1))
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Dropout(dropout_ratio))
        self.classifier.append(MaskedConv1D(in_channels = embed_dim, out_channels = embed_dim, 
        kernel_size = 3, stride=1,padding=1,bias=True))
        self.classifier.append(nn.LeakyReLU(0.2))
        self.classifier.append(nn.Dropout(0.7))
        self.classifier.append(MaskedConv1D(in_channels = embed_dim, out_channels = n_class+1, 
        kernel_size = 1, stride=1,padding=0,bias=True))
        # self.cadl = CADL()
        # self.attention = Non_Local_Block(embed_dim,mid_dim,dropout_ratio)
        
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
        self.batch_avg=nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        
        self.apply(weights_init)

    def forward(self, inputs,audio_feature, is_training=True, **args):
        # features = inputs.transpose(-1, -2)
        # b,c,n=features.size()
        # import pdb
        audio_feat = audio_feature
        features,batched_masks_v = self.preprocessing(inputs,training = is_training)
        features = features.to(self.classifier[1].conv.weight.device)
        batched_masks_v = batched_masks_v.to(self.classifier[1].conv.weight.device)
        # pdb.set_trace()
        # print('1',features.device)
        batched_masks_f = batched_masks_v
        feat_vs = features[:,:1024,:]
        feat_fs = features[:,1024:,:]
        feat_vs,mask_v = self.backbone_v(feat_vs,batched_masks_v)
        feat_fs,mask_f = self.backbone_f(feat_fs,batched_masks_f)

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
        # vfeat,vmask = self.neck_v(feat_vs,vmask)
        # ffeat,fmask = self.neck_f(feat_fs,fmask)

        # v_atn = self.attention_v(feat_vs)
        # f_atn = self.attention_f(feat_fs)
        # v_atn = self.atn_head_v(vfeat,vmask)
        # f_atn = self.atn_head_f(ffeat,fmask)
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
        # pdb.set_trace()
        # print('6',x_cls.device)
        # x_cls = self.cls_head(nfeat,vmask)
        # fg_mask, bg_mask,dropped_fg_mask = self.cadl(x_cls, x_atn, include_min=True)
        return {'feat':nfeat.transpose(-1, -2), 'cas':x_cls.transpose(-1, -2), 'attn':x_atn.transpose(-1, -2),\
            # 'v_atn_orig':v_atn_orig,'f_atn_orig':f_atn_orig,\
            'v_atn':v_atn.transpose(-1, -2),'f_atn':f_atn.transpose(-1, -2),'mask':mask_f.transpose(-1, -2)}
            #,fg_mask.transpose(-1, -2), bg_mask.transpose(-1, -2),dropped_fg_mask.transpose(-1, -2)
        # return att_sigmoid,att_logit, feat_emb, bag_logit, instance_logit

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn,mask= outputs['feat'],outputs['cas'],outputs['attn'],outputs['mask']
        
        #element_logits：分类概率
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        element_logits_supp_total = self._multiply(element_logits, element_atn,include_min=True)
        loss_3_supp_Contrastive = self.Contrastive(feat,element_logits_supp_total,labels,mask,is_back=False)
        # print(feat.shape, element_logits.shape, element_atn.shape,mask.shape,v_atn.shape,f_atn.shape)
        # torch.Size([6, 2304, 2048]) torch.Size([6, 2304, 21]) torch.Size([6, 2304, 1]) torch.Size([6, 2304, 1]) 
        # torch.Size([6, 2304, 1]) torch.Size([6, 2304, 1])
        # import pdb;pdb.set_trace();
        b,n,c = feat.shape
        v_atn = v_atn*mask
        f_atn = f_atn*mask
        feat = feat*mask
        element_logits = element_logits*mask
        element_atn = element_atn*mask

        loss_mil_orig_total,loss_mil_supp_total,mutual_loss_total,\
        loss_norm_total,v_loss_norm_total,f_loss_norm_total,loss_guide_total,v_loss_guide_total,f_loss_guide_total \
            = [torch.zeros(1) for i in range(9)]
        
        
        #!!!!!!!!!!!!!mask!!!!!!!!!!!!!
        for i in range(b):
            true_length = len(mask[i][mask[i]!=0])
            v_atn_x = v_atn[i][:true_length,:].view(1,-1,1)
            f_atn_x = f_atn[i][:true_length,:].view(1,-1,1)
            element_logits_x = element_logits[i][:true_length,:].view(1,-1,21)
            feat_x = feat[i][:true_length,:].view(1,-1,2048)
            element_atn_x = element_atn[i][:true_length,:].view(1,-1,1)

            # import pdb;pdb.set_trace();
            
            mutual_loss = 0.5*F.mse_loss(v_atn_x,f_atn_x.detach())+0.5*F.mse_loss(f_atn_x,v_atn_x.detach())
            mutual_loss_total+=mutual_loss
            #detach：返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
            #不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
            #learning weight dynamic, lambda1 (1-lambda1)

            element_logits_supp = self._multiply(element_logits_x, element_atn_x,include_min=True)
            loss_mil_orig, _ = self.topkloss(element_logits_x,
                                        labels[i].reshape(1,20),
                                        is_back=True,
                                        rat=args['opt'].k,
                                        reduce=None)
            loss_mil_orig_total += loss_mil_orig.mean()
            # SAL
            loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                                labels[i].reshape(1,20),
                                                is_back=False,
                                                rat=args['opt'].k,
                                                reduce=None)
            loss_mil_supp_total += loss_mil_supp.mean()
                    

            loss_norm = element_atn_x.mean()
            loss_norm_total+=loss_norm
            # guide loss，Loppo
            loss_guide = (1 - element_atn_x -
                        element_logits_x.softmax(-1)[..., [-1]]).abs().mean()
                        #element_logits.softmax(-1)[..., [-1]]：Sc+1
            loss_guide_total+=loss_guide
            v_loss_norm = v_atn_x.mean()
            v_loss_norm_total+=v_loss_norm
            # guide loss
            v_loss_guide = (1 - v_atn_x -
                        element_logits_x.softmax(-1)[..., [-1]]).abs().mean()
            v_loss_guide_total+=v_loss_guide

            f_loss_norm = f_atn_x.mean()
            f_loss_norm_total+=f_loss_norm
            # guide loss
            f_loss_guide = (1 - f_atn_x -
                        element_logits_x.softmax(-1)[..., [-1]]).abs().mean()
            f_loss_guide_total+=f_loss_guide

        # total loss
        # total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean() +
        #               args['opt'].alpha3*loss_3_supp_Contrastive+
        #               args['opt'].alpha4*mutual_loss+
        #               args['opt'].alpha1*(loss_norm+v_loss_norm+f_loss_norm)/3 +
        #               args['opt'].alpha2*(loss_guide+v_loss_guide+f_loss_guide)/3)

        loss_mil_orig_total/=b
        loss_mil_supp_total/=b
        mutual_loss_total/=b
        loss_norm_total/=b
        v_loss_norm_total/=b
        f_loss_norm_total/=b
        loss_guide_total/=b
        v_loss_guide_total/=b
        f_loss_guide_total/=b

        total_loss = (loss_mil_orig_total + loss_mil_supp_total +
                        args['opt'].alpha3*loss_3_supp_Contrastive+
                        args['opt'].alpha4*mutual_loss_total+
                        args['opt'].alpha1*(loss_norm_total+v_loss_norm_total+f_loss_norm_total)/3 +
                        args['opt'].alpha2*(loss_guide_total+v_loss_guide_total+f_loss_guide_total)/3)
        # output = torch.cosine_similarity(dropped_fg_feat, fg_feat, dim=1)
        # pdb.set_trace()

        return total_loss,loss_mil_orig_total,loss_mil_supp_total,loss_3_supp_Contrastive,mutual_loss_total,\
            (loss_norm_total+v_loss_norm_total+f_loss_norm_total)/3,(loss_guide_total+v_loss_guide_total+f_loss_guide_total)/3
    
    def criterion_copy(self, outputs, labels, **args):
        feat, element_logits, element_atn= outputs['feat'],outputs['cas'],outputs['attn']
        #element_logits：分类概率
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        mutual_loss=0.5*F.mse_loss(v_atn,f_atn.detach())+0.5*F.mse_loss(f_atn,v_atn.detach())
        #detach：返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
        #不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        #learning weight dynamic, lambda1 (1-lambda1) 
        b,n,c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn,include_min=True)
        
        loss_mil_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=args['opt'].k,
                                       reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)
        # import pdb;pdb.set_trace();
        loss_3_supp_Contrastive = self.Contrastive(feat,element_logits_supp,labels,is_back=False)
        

        loss_norm = element_atn.mean()
        # guide loss，Loppo
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()
                      #element_logits.softmax(-1)[..., [-1]]：Sc+1

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()
        
        # total loss
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean() +
                      args['opt'].alpha3*loss_3_supp_Contrastive+
                      args['opt'].alpha4*mutual_loss+
                      args['opt'].alpha1*(loss_norm+v_loss_norm+f_loss_norm)/3 +
                      args['opt'].alpha2*(loss_guide+v_loss_guide+f_loss_guide)/3)
       
        # output = torch.cosine_similarity(dropped_fg_feat, fg_feat, dim=1)
        # pdb.set_trace()

        return total_loss,loss_mil_orig.mean(),loss_mil_supp.mean(),loss_3_supp_Contrastive,mutual_loss,\
            (loss_norm+v_loss_norm+f_loss_norm)/3,(loss_guide+v_loss_guide+f_loss_guide)/3

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)#(10,20)+(10,1)=(10,21)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)#(10,40,21);k为mil中bag中的元素个数
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self,x,element_logits,labels,mask,is_back=False):#Co-Activity Similarity
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        # _, n, c = element_logits.shape

        #x:10,320,2048,element_logits:10,320,21
        for i in range(0, 3*2, 2):
            true_length1 = len(mask[i][mask[i]!=0])
            true_length2 = len(mask[i+1][mask[i+1]!=0])
            atn1 = F.softmax(element_logits[i][:true_length1,:], dim=0)#320,21
            atn2 = F.softmax(element_logits[i+1][:true_length2,:], dim=0)

            n1 = torch.FloatTensor([np.maximum(true_length1-1, 1)]).cuda()#tensor([319.])
            n2 = torch.FloatTensor([np.maximum(true_length2-1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i][:true_length1,:], 1, 0), atn1)      # (n_feature, n_class)(2048,21)
            Hf2 = torch.mm(torch.transpose(x[i+1][:true_length2,:], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i][:true_length1,:], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[i+1][:true_length2,:], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity (n_class)
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        sim_loss = sim_loss / n_tmp
        return sim_loss
    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn   = outputs
        
        return element_logits,element_atn
    
    @torch.no_grad()
    def preprocessing(self,video_list, training = True,padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x.transpose(-1,-2) for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            # if max_len <= max_seq_len:
            #     max_len = max_seq_len
            # else:
            #     # pad the input to the next divisible size
            #     stride = max_div_factor
            #     max_len = (max_len + (stride - 1)) // stride * stride
            # padding_size = [0, max_len - feats_lens[0]]
            # batched_inputs = F.pad(
            #     feats[0], padding_size, value=padding_val).unsqueeze(0)
            batched_inputs = feats[0].unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs
        batched_masks = batched_masks.unsqueeze(1)
        return batched_inputs, batched_masks

class ANT_CO2(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim=2048
        mid_dim=1024
        dropout_ratio=args['opt'].dropout_ratio
        reduce_ratio=args['opt'].reduce_ratio

        self.vAttn = getattr(model,args['opt'].AWM)(1024,args)
        self.fAttn = getattr(model,args['opt'].AWM)(1024,args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7), 
            nn.Conv1d(embed_dim, n_class+1, 1))
        # self.cadl = CADL()
        # self.attention = Non_Local_Block(embed_dim,mid_dim,dropout_ratio)
        
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
        self.batch_avg=nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        _kernel = ((args['opt'].max_seqlen // args['opt'].t) // 2 * 2 + 1)
        self.pool=nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()
        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b,c,n=feat.size()
        # feat = self.feat_encoder(x)
        v_atn,vfeat = self.vAttn(feat[:,:1024,:],feat[:,1024:,:])
        f_atn,ffeat = self.fAttn(feat[:,1024:,:],feat[:,:1024,:])
        x_atn = (f_atn+v_atn)/2
        nfeat = torch.cat((vfeat,ffeat),1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)
        x_cls=self.pool(x_cls)
        x_atn=self.pool(x_atn)
        f_atn=self.pool(f_atn)
        v_atn=self.pool(v_atn)
        # fg_mask, bg_mask,dropped_fg_mask = self.cadl(x_cls, x_atn, include_min=True)

        return {'feat':nfeat.transpose(-1, -2), 'cas':x_cls.transpose(-1, -2), 'attn':x_atn.transpose(-1, -2), 'v_atn':v_atn.transpose(-1, -2),'f_atn':f_atn.transpose(-1, -2)}
            #,fg_mask.transpose(-1, -2), bg_mask.transpose(-1, -2),dropped_fg_mask.transpose(-1, -2)
        # return att_sigmoid,att_logit, feat_emb, bag_logit, instance_logit


    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn= outputs['feat'],outputs['cas'],outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        mutual_loss=0.5*F.mse_loss(v_atn,f_atn.detach())+0.5*F.mse_loss(f_atn,v_atn.detach())
        #learning weight dynamic, lambda1 (1-lambda1) 
        b,n,c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn,include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=args['opt'].k,
                                       reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)
        
        loss_3_supp_Contrastive = self.Contrastive(feat,element_logits_supp,labels,is_back=False)
        

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean() + args['opt'].alpha3*loss_3_supp_Contrastive +mutual_loss+
                      args['opt'].alpha1*(loss_norm+v_loss_norm+f_loss_norm)/3 +
                      args['opt'].alpha2*(loss_guide+v_loss_guide+f_loss_guide)/3)
       
        # output = torch.cosine_similarity(dropped_fg_feat, fg_feat, dim=1)
        # pdb.set_trace()

        return total_loss

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self,x,element_logits,labels,is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3*2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i+1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)      # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        sim_loss = sim_loss / n_tmp
        return sim_loss
    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn   = outputs
        
        return element_logits,element_atn

class CO3(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim=2048
        mid_dim=1024
        audio_dim=128
        dropout_ratio=args['opt'].dropout_ratio
        reduce_ratio=args['opt'].reduce_ratio

        self.vAttn = AV_Fuison()
        self.fAttn = AV_Fuison()
        self.aAttn = getattr(model,args['opt'].AWM)(1024,args)
        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(audio_dim, mid_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0),nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(n_feature, n_feature, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(n_feature, n_class+1, 1))
        # self.cadl = CADL()
        # self.attention = Non_Local_Block(embed_dim,mid_dim,dropout_ratio)
        
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
        self.batch_avg=nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        
        self.apply(weights_init)

    def forward(self, inputs,audio_feature, is_training=True, **args):
        feat = inputs
        audio_feat = audio_feature.transpose(-1, -2)
        afeat = self.audio_encoder(audio_feat) 
        afeat = afeat.transpose(-1,-2)
        b,c,n=feat.size()
        # feat = self.feat_encoder(x)
        v_atn,vfeat = self.vAttn(afeat,feat[:,:,:1024])
        f_atn,ffeat = self.fAttn(afeat,feat[:,:,1024:])
        x_atn = (f_atn+v_atn)/2
        nfeat = torch.cat((vfeat,ffeat),1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)

        # fg_mask, bg_mask,dropped_fg_mask = self.cadl(x_cls, x_atn, include_min=True)

        return {'feat':nfeat.transpose(-1, -2), 'cas':x_cls.transpose(-1, -2), 'attn':x_atn.transpose(-1, -2), 'v_atn':v_atn.transpose(-1, -2),'f_atn':f_atn.transpose(-1, -2)}
            #,fg_mask.transpose(-1, -2), bg_mask.transpose(-1, -2),dropped_fg_mask.transpose(-1, -2)
        # return att_sigmoid,att_logit, feat_emb, bag_logit, instance_logit


    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn= outputs['feat'],outputs['cas'],outputs['attn']
        #element_logits：分类概率
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        mutual_loss=0.5*F.mse_loss(v_atn,f_atn.detach())+0.5*F.mse_loss(f_atn,v_atn.detach())
        #detach：返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
        #不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        #learning weight dynamic, lambda1 (1-lambda1) 
        b,n,c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn,include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                       labels,
                                       is_back=True,
                                       rat=args['opt'].k,
                                       reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                            labels,
                                            is_back=False,
                                            rat=args['opt'].k,
                                            reduce=None)
        
        loss_3_supp_Contrastive = self.Contrastive(feat,element_logits_supp,labels,is_back=False)
        

        loss_norm = element_atn.mean()
        # guide loss，Loppo
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()
                      #element_logits.softmax(-1)[..., [-1]]：Sc+1

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()
        
        # total loss
        total_loss = (loss_mil_orig.mean() + loss_mil_supp.mean() +
                      args['opt'].alpha3*loss_3_supp_Contrastive+
                      args['opt'].alpha4*mutual_loss+
                      args['opt'].alpha1*(loss_norm+v_loss_norm+f_loss_norm)/3 +
                      args['opt'].alpha2*(loss_guide+v_loss_guide+f_loss_guide)/3)
       
        # output = torch.cosine_similarity(dropped_fg_feat, fg_feat, dim=1)
        # pdb.set_trace()

        return total_loss

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):
        
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)#(10,20)+(10,1)=(10,21)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)#(10,40,21);k为mil中bag中的元素个数
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self,x,element_logits,labels,is_back=False):#Co-Activity Similarity
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        #x:10,320,2048
        for i in range(0, 3*2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)#320,21
            atn2 = F.softmax(element_logits[i+1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()#1
            n2 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)      # (n_feature, n_class)(2048,21)
            Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity (n_class)
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        sim_loss = sim_loss / n_tmp
        return sim_loss
    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn   = outputs
        
        return element_logits,element_atn
