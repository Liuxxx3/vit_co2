import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
import time

from tensorboard_logger import Logger

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train(itr, dataset, args, model, optimizer, logger, device):
    model.train()
    features,audio_features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]
    audio_features = audio_features[:,:np.max(seq_len),:]
    
    features = torch.from_numpy(features).float().to(device)
    audio_features = torch.from_numpy(audio_features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    #interative
    pseudo_label = None
   
    outputs = model(features,audio_features,seq_len=seq_len,is_training=True,itr=itr,opt=args)
    
    total_loss,loss_mil_orig,loss_mil_supp,loss_3_supp_Contrastive,mutual_loss,loss_norm,loss_guide \
        = model.criterion(outputs,labels,seq_len=seq_len,device=device,logger=logger,opt=args,itr=itr,pairs_id=pairs_id,inputs=features)
    # print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))
    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)
    total_loss.backward()
    optimizer.step()
    return total_loss.data.cpu().numpy(),loss_mil_orig.data.cpu().numpy(),loss_mil_supp.data.cpu().numpy(),\
        loss_3_supp_Contrastive.data.cpu().numpy(),mutual_loss,loss_norm.data.cpu().numpy(),loss_guide.data.cpu().numpy()

