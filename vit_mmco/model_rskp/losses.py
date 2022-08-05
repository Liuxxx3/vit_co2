from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class NormalizedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels):
        new_labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-8)
        loss = -1.0 * torch.mean(torch.sum(Variable(new_labels) * torch.log(pred), dim=1), dim=0)
        return loss


class CategoryCrossEntropy(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, pred, soft_label):
        #pred,soft_label:(ntc)
        soft_label = F.softmax(soft_label / self.T, -1)
        soft_label = Variable(soft_label.detach().data, requires_grad=False)
        loss = -1.0 * torch.sum(Variable(soft_label) * torch.log_softmax(pred / self.T, -1), dim=-1)
        loss = loss.mean(-1).mean(-1)
        return loss


class AttLoss(nn.Module):
    def __init__(self, s_factor):
        super().__init__()
        self.s = s_factor

    def forward(self, att):
        t = att.size(1)
        max_att_values, _ = torch.topk(att, max(int(t // self.s), 1), -1)
        mean_max_att = max_att_values.mean(1)

        min_att_values, _ = torch.topk(-att, max(int(t // self.s), 1), -1)
        mean_min_att = -min_att_values.mean(1)

        loss = (mean_min_att - mean_max_att).mean(0)

        return loss

class Co2_Loss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.k = 7.
        self.alpha1 = 0.8
        self.alpha2 = 0.8
        self.alpha3 = 1.0
        self.alpha4 = 1.0
        self.device = torch.device(
            'cuda:' + str(args.gpu_ids[0]) if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu')

    def forward(self,outputs,labels):
        feat, element_logits, element_atn,mask= outputs['feat'],outputs['cas'],outputs['attn'],outputs['mask']
        
        #element_logits：分类概率
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        element_logits_supp_total = self._multiply(element_logits, element_atn,include_min=True)
        # import pdb;pdb.set_trace();
        loss_3_supp_Contrastive = self.Contrastive(feat,element_logits_supp_total,labels,mask,is_back=False)
        # print(feat.shape, element_logits.shape, element_atn.shape,mask.shape,v_atn.shape,f_atn.shape)
        # torch.Size([6, 2304, 2048]) torch.Size([6, 2304, 21]) torch.Size([6, 2304, 1]) torch.Size([6, 2304, 1]) 
        # torch.Size([6, 2304, 1]) torch.Size([6, 2304, 1])

        b,t,c = feat.shape
        v_atn = v_atn*mask
        f_atn = f_atn*mask
        feat = feat*mask
        element_logits = element_logits*mask
        element_atn = element_atn*mask

        loss_mil_orig_total,loss_mil_supp_total,mutual_loss_total,\
        loss_norm_total,v_loss_norm_total,f_loss_norm_total,loss_guide_total,v_loss_guide_total,f_loss_guide_total \
            = [torch.zeros(1) for i in range(9)]
        loss_mil_orig_total = loss_mil_orig_total.to(self.device)
        loss_mil_supp_total = loss_mil_supp_total.to(self.device)
        mutual_loss_total = mutual_loss_total.to(self.device)
        loss_norm_total = loss_norm_total.to(self.device)
        v_loss_norm_total = v_loss_norm_total.to(self.device)
        f_loss_norm_total = f_loss_norm_total.to(self.device)
        loss_guide_total = loss_guide_total.to(self.device)
        v_loss_guide_total = v_loss_guide_total.to(self.device)
        f_loss_guide_total = f_loss_guide_total.to(self.device)
        #!!!!!!!!!!!!!mask!!!!!!!!!!!!!
        for i in range(b):
            true_length = len(mask[i][mask[i]!=0])
            v_atn_x = v_atn[i][:true_length,:].view(1,-1,1)
            f_atn_x = f_atn[i][:true_length,:].view(1,-1,1)
            element_logits_x = element_logits[i][:true_length,:].view(1,-1,21)
            feat_x = feat[i][:true_length,:].view(1,-1,2048)
            element_atn_x = element_atn[i][:true_length,:].view(1,-1,1)

            
            mutual_loss = 0.5*F.mse_loss(v_atn_x,f_atn_x.detach())+0.5*F.mse_loss(f_atn_x,v_atn_x.detach())
            mutual_loss_total+=mutual_loss
            #detach：返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
            #不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
            #learning weight dynamic, lambda1 (1-lambda1)

            element_logits_supp = self._multiply(element_logits_x, element_atn_x,include_min=True)
            loss_mil_orig, _ = self.topkloss(element_logits_x,
                                        labels[i].reshape(1,20),
                                        is_back=True,
                                        rat=self.k,
                                        reduce=None)
            loss_mil_orig_total += loss_mil_orig.mean()
            # SAL
            loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                                labels[i].reshape(1,20),
                                                is_back=False,
                                                rat=self.k,
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
        divisor = 10
        loss_mil_orig_total/=divisor
        loss_mil_supp_total/=divisor
        mutual_loss_total/=divisor
        loss_norm_total/=divisor
        v_loss_norm_total/=divisor
        f_loss_norm_total/=divisor
        loss_guide_total/=divisor
        v_loss_guide_total/=divisor
        f_loss_guide_total/=divisor
        
        total_loss = (loss_mil_orig_total + loss_mil_supp_total +
                        self.alpha3*loss_3_supp_Contrastive+
                        self.alpha4*mutual_loss_total+
                        self.alpha1*(loss_norm_total+v_loss_norm_total+f_loss_norm_total)/3 +
                        self.alpha2*(loss_guide_total+v_loss_guide_total+f_loss_guide_total)/3)
        # output = torch.cosine_similarity(dropped_fg_feat, fg_feat, dim=1)
        # pdb.set_trace()

        return total_loss,loss_mil_orig_total,loss_mil_supp_total,loss_3_supp_Contrastive,mutual_loss_total,\
            (loss_norm_total+v_loss_norm_total+f_loss_norm_total)/3,(loss_guide_total+v_loss_guide_total+f_loss_guide_total)/3
            
    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min
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