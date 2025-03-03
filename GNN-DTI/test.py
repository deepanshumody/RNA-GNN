import pickle
from gnn import gnn,FocalLoss
import time
import numpy as np
import utils
import torch.nn as nn
import torch
import time
import os
from sklearn.metrics import roc_auc_score
import argparse
import time
from torch.utils.data import DataLoader                                     
from dataset import collate_fn, DTISampler
import glob
import random
import pandas as pd
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
file1 = open('nonredundantRNA.txt', 'r')
Lines = file1.readlines()
file1.close()

data_list=[]

for i in ['./4RUM.pkl']:
	with open(i, 'rb') as f:
	    newdict = pickle.load(f)
	if(len(newdict)==0):
		continue
	for coord,graph in ((newdict.items())):
		graph['key']=coord+i[-12:-8]
		data_list.append(graph)
		#print(graph.keys())
	#if(len(data_list)>31):
		#break


parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 1000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 16)
parser.add_argument("--num_workers", help="number of workers", type=int, default = 0)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
#parser.add_argument("--dude_data_fpath", help="file path of dude data", type=str, default='data/')
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default = './savemorepos/')
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.3)
#parser.add_argument("--train_keys", help="train keys", type=str, default='keys/train_keys.pkl')
#parser.add_argument("--test_keys", help="test keys", type=str, default='keys/test_keys.pkl')
args = parser.parse_args()
print (args)

#hyper parameters
num_epochs = args.epoch
lr = args.lr
ngpu = args.ngpu
batch_size = args.batch_size
#dude_data_fpath = args.dude_data_fpath
save_dir = args.save_dir

#make save dir if it doesn't exist
if not os.path.isdir(save_dir):
    os.system('mkdir ' + save_dir)

#read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.
#with open (args.train_keys, 'rb') as fp:
    #train_keys = pickle.load(fp)
#with open (args.test_keys, 'rb') as fp:
    #test_keys = pickle.load(fp)

#print simple statistics about dude data and pdbbind data
#print (f'Number of train data: {len(train_keys)}')
#print (f'Number of test data: {len(test_keys)}')

#initialize model
#if args.ngpu>0:
    #cmd = utils.set_cuda_visible_device(args.ngpu)
    #os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
model = gnn(args)
print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, load_save_file='/home/kihara/modyd/Desktop/RNA_GNN_Nov11/model_GNN_DTI/savemorepos/save_172.pt')
print('Device: {}'.format(device))
#train and test dataset
                     
test_dataloader = DataLoader(data_list, args.batch_size, \
     shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#loss function
loss_fn = nn.BCELoss()
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

st = time.time()
#collect losses of each iteration

test_losses = [] 

#collect true label of each iteration

test_true = []
ke=[]
#collect predicted label of each iteration

test_pred = []

model.eval()
for i_batch, sample in enumerate(test_dataloader):
	model.zero_grad()
	H, A1, A2, Y, V, keys = sample 
	H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device),\
		          Y.to(device), V.to(device)

	#train neural network
	pred = model.train_model((H, A1, A2, V))

	loss = loss_fn(pred, Y) 

	#collect loss, true label and predicted label
	test_losses.append(loss.data.cpu().numpy())
	test_true.append(Y.data.cpu().numpy())
	test_pred.append(pred.data.cpu().numpy())
	ke=ke+(keys)
	
	#print(keys)
	#if i_batch>10 : break


test_losses = np.mean(np.array(test_losses))

test_pred = np.concatenate(np.array(test_pred), 0)

test_true = np.concatenate(np.array(test_true), 0)
print(len(ke))
print(test_pred.shape)
data={}
data['y_pred'] = test_pred.reshape(-1)
data['y_true'] = test_true.reshape(-1)
data['coord'] = np.array(ke).reshape(-1)
df = pd.DataFrame(data=data)
df.to_csv('preds_RNA_DTImorepos' + '.csv', sep=',', index=False)

test_roc = roc_auc_score(test_true, test_pred) 
end = time.time()
print (test_roc)
