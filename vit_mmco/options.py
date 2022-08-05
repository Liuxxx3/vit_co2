import argparse

parser = argparse.ArgumentParser(description='CO2-NET')
parser.add_argument('--path-dataset', type=str, default='Thumos14', help='the path of data feature')
parser.add_argument('--lr', type=float, default=0.00001,help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=6, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model-name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num-class', type=int,default=20, help='number of classes (default: )')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on (default: )')
parser.add_argument('--max_seqlen', type=int, default=320, help='maximum sequence length during training (default: 750)')
parser.add_argument('--num-similar', default=3, type=int,help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--seed', type=int, default=3552, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=20000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used I3D or UNT (default: I3D)')
parser.add_argument('--use-model',type=str,help='model used to train the network')
parser.add_argument('--interval', type=int, default=20,help='time interval of performing the test')
parser.add_argument('--similar-size', type=int, default=2)

parser.add_argument('--mu-num', type=int, default=8, help='number of Gaussians')
parser.add_argument('--max-len', default=2304, type=int,help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--em-iter', type=int, default=2, help='number of EM iteration')
parser.add_argument('--T', type=float, default=0.2, help='number of head')
parser.add_argument('--lambda-s', default=1.0, help='weight of pseudo label supervision loss')
parser.add_argument('--warmup-epoch', default=2000, help='epoch starting to use the inter-video branch')
parser.add_argument('--mu-queue-len', type=int, default=5, help='number of slots of each class of memory bank')
parser.add_argument('--inp-feat-num', type=int, default=2048, help='size of input feature (default: 2048)')
parser.add_argument('--out-feat-num', type=int, default=2048, help='size of output feature (default: 2048)')
parser.add_argument('--gpus', type=int, default=[1], nargs='+', help='used gpu')
parser.add_argument('--w', type=float, default=0.5, help='number of head')

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dataset',type=str,default='SampleDataset')
parser.add_argument('--proposal_method',type=str,default='multiple_threshold_hamnet')

#for proposal genration
parser.add_argument('--scale',type=float,default=1)
parser.add_argument("--feature_fps", type=int, default=25)
parser.add_argument('--gamma-oic', type=float, default=0.2)


parser.add_argument('--k',type=float,default=7)
# for testing time usage
parser.add_argument("--topk2", type=float, default=10)
parser.add_argument("--topk", type=float, default=60)


parser.add_argument('--dropout_ratio',type=float,default=0.7)
parser.add_argument('--reduce_ratio',type=int,default=16)
# for pooling kernel size calculate
parser.add_argument('--t',type=int,default=5)



#-------------loss weight---------------
parser.add_argument("--alpha1", type=float, default=0.8)
parser.add_argument("--alpha2", type=float, default=0.8)
parser.add_argument("--alpha3", type=float, default=1)
parser.add_argument('--alpha4',type=float,default=1)


parser.add_argument("--AWM", type=str, default='BWA_fusion_dropout_feat_v2')

