from __future__ import print_function
import argparse
import os
from platform import architecture
import torch
import torch.nn as nn
import model
import multiprocessing as mp
import wsad_dataset

import random
from test import test
from train import train
from tensorboard_logger import Logger
# from torch.utils.tensorboard import SummaryWriter

import options
import numpy as np
from torch.optim import lr_scheduler
from tqdm import tqdm
import shutil
from model_rskp.memory import Memory
from model_rskp.losses import CategoryCrossEntropy,Co2_Loss
from eval.eval import ft_eval

torch.set_default_tensor_type('torch.cuda.FloatTensor')
def setup_seed(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
import torch.optim as optim
def visible_gpu(gpus):
    """
        set visible gpu.
        can be a single id, or a list
        return a list of new gpus ids
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, gpus)))
    return list(range(len(gpus)))

if __name__ == '__main__':
   pool = mp.Pool(5)

   args = options.parser.parse_args()
   # seed = random.randint(1,10000)
   seed=args.seed
   print('=============seed: {}, pid: {}============='.format(seed,os.getpid()))
   setup_seed(seed)
   # torch.manual_seed(args.seed)
   device = torch.device("cuda")
   dataset = getattr(wsad_dataset,args.dataset)(args)
   if 'Thumos' in args.dataset_name:
      max_map=[0]*9
   else:
      max_map=[0]*10
   if not os.path.exists('./ckpt/'):
      os.makedirs('./ckpt/')
   if not os.path.exists('./logs/' + args.model_name):
      os.makedirs('./logs/' + args.model_name)
   if os.path.exists('./logs/' + args.model_name):
      shutil.rmtree('./logs/' + args.model_name)
   logger = Logger('./logs/' + args.model_name)
#    writer = SummaryWriter('./logs' + args.model_name)
   print(args)
   # model = Model(dataset.feature_size, dataset.num_class).to(device)
   model = getattr(model,args.use_model)(dataset.feature_size, dataset.num_class,opt=args)
   args.gpu_ids = visible_gpu(args.gpus)
   memory = Memory(args).to(device)
   loss_spl = CategoryCrossEntropy(args.T)
   co2_loss = Co2_Loss(args)
   
#    device_ids = [1]
   model = nn.DataParallel(model).to(device)
   if args.pretrained_ckpt is not None:
      model.load_state_dict(torch.load(args.pretrained_ckpt))

#    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   optimizer = optim.AdamW(model.parameters(), lr=args.lr*(torch.cuda.device_count()), weight_decay=args.weight_decay)
   # optimizer = optim.SGD(model.parameters(), lr=args.lr,
                  #   momentum=args.momentum, weight_decay=args.weight_decay)
   # scheduler = lr_scheduler.StepLR(optimizer,step_size = args.lr_decay,gamma = 0.5)

   total_loss = 0
   lrs = [args.lr, args.lr/5, args.lr/5/5]
   print(model)
   for itr in tqdm(range(args.max_iter)):
      loss,loss_mil_orig,loss_mil_supp,loss_3_supp_Contrastive,mutual_loss,loss_norm,loss_guide,vid_spl_loss,mem \
          = train(itr, memory,co2_loss,loss_spl,dataset, args, model, optimizer, logger, device)
      memory = mem
      total_loss+=loss
      info = { 'total_loss': total_loss/args.interval,
               'loss_mil_orig':loss_mil_orig,
               'loss_mil_supp':loss_mil_supp,
               'loss_3_supp_Contrastive':loss_3_supp_Contrastive,
               'mutual_loss':mutual_loss,
               'loss_norm':loss_norm,
               'loss_guide':loss_guide,
               'vid_spl_loss':vid_spl_loss
               }
      
      for tag,value in info.items():
        logger.log_value(tag, value, itr)

      if itr == args.warmup_epoch:
         model.eval()
         mu_queue, sc_queue, lbl_queue = ft_eval(dataset, model, args, device)
         memory._init_queue(mu_queue, sc_queue, lbl_queue)
         # import pdb;pdb.set_trace();
         model.train()
         args.lambda_s = 0.5

      if itr % args.interval == 0 and not itr == 0:
         print('Iteration: %d, Loss: %.5f' %(itr, total_loss/args.interval))
         total_loss = 0
         torch.save(model.state_dict(), './ckpt/last_' + args.model_name + '.pkl')
         iou,dmap = test(itr, dataset, args, model, logger, device,pool)
         if 'Thumos' in args.dataset_name:
            cond=sum(dmap[:7])>sum(max_map[:7])
         else:
            cond=np.mean(dmap)>np.mean(max_map)
         if cond:
            torch.save(model.state_dict(), './ckpt/best_' + args.model_name + '.pkl')
            max_map = dmap


         print('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i],max_map[i]*100) for i in range(len(iou))]))
         max_map = np.array(max_map)
         print('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(max_map[:5])*100,np.mean(max_map[:7])*100,np.mean(max_map)*100))
         print("------------------pid: {}--------------------".format(os.getpid()))

    
