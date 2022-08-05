import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
from model import random_walk
from tensorboard_logger import Logger
from model_rskp.losses import AttLoss,Co2_Loss
from model_rskp.memory import Memory
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train(itr,memory,co2_loss,loss_spl,dataset, args, model: torch.nn.DataParallel, optimizer, logger, device):
    model.train()
    features,audio_features, np_labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    # seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    # features = features[:,:np.max(seq_len),:]
    # audio_features = audio_features[:,:np.max(seq_len),:]
    
    # features = torch.from_numpy(features).float().to(device)
    for i in range(0,features.shape[0]):
        features[i] = torch.from_numpy(features[i]).float().to(device)
    labels = torch.from_numpy(np_labels).float().to(device)
    #interative
    pseudo_label = None
   
    o_output,m_output,em_out,mask = model(features,is_training=True,itr=itr,opt=args)
    # import pdb; pdb.set_trace()
    total_loss_o,loss_mil_orig_o,loss_mil_supp_o,loss_3_supp_Contrastive_o,mutual_loss_o,loss_norm_o,loss_guide_o \
        = co2_loss(o_output,labels)
    total_loss_m,loss_mil_orig_m,loss_mil_supp_m,loss_3_supp_Contrastive_m,mutual_loss_m,loss_norm_m,loss_guide_m \
        = co2_loss(m_output,labels)
    vid_co2_loss = total_loss_o +total_loss_m
    if itr > args.warmup_epoch:
        reallocated_xs = []
        idxs = []
        for np_label in  np_labels:
            idx = np.where(np_label==1)[0].tolist()
            idxs.append(idx)
            cls_mu = memory._return_queue(idx).detach()
            reallocated_x = random_walk(em_out[0], cls_mu, args.w,mask)
            reallocated_xs.append(reallocated_x)
        reallocated_xs = torch.stack(reallocated_xs).squeeze(1)
        output_reallocated_x = model.module.PredictionModule(reallocated_xs,mask,mask)
        vid_co2_loss += co2_loss(output_reallocated_x,labels)[0]
        # vid_fore_loss += 0.5 * self.loss_nce(r_vid_ca_pred, f_labels)
        # vid_back_loss += 0.5 * self.loss_nce(r_vid_cw_pred, b_labels)
        vid_spl_loss = loss_spl(o_output['cas'], output_reallocated_x['cas'] * 0.2 + m_output['cas'] * 0.8)
        for em_out_mu,em_out_mu_pred,idx in zip(em_out[1],em_out[2],idxs):
            # memory._update_queue(em_out[1].squeeze(0), em_out[2].squeeze(0), idxs)
            memory._update_queue(em_out_mu,em_out_mu_pred, idx)
    else:
        vid_spl_loss = loss_spl(o_output['cas'], m_output['cas'])
    # print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))
    total_loss = vid_spl_loss * args.lambda_s + vid_co2_loss
    loss_mil_orig = loss_mil_orig_m + loss_mil_orig_o
    loss_mil_supp = loss_mil_supp_m + loss_mil_supp_o
    loss_3_supp_Contrastive = loss_3_supp_Contrastive_m + loss_3_supp_Contrastive_o
    mutual_loss = mutual_loss_m + mutual_loss_o
    loss_norm = loss_norm_m + loss_norm_o
    loss_guide = loss_guide_m + loss_guide_o
    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)
    total_loss.backward()
    optimizer.step()
    return total_loss.data.cpu().numpy(),loss_mil_orig.data.cpu().numpy(),loss_mil_supp.data.cpu().numpy(),\
        loss_3_supp_Contrastive.data.cpu().numpy(),mutual_loss,loss_norm.data.cpu().numpy(),loss_guide.data.cpu().numpy(),\
        vid_spl_loss.data.cpu().numpy(),memory

