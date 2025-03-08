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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import argparse
import time
from torch.utils.data import DataLoader                                     
from dataset import collate_fn, DTISampler
import glob
import random
import pandas as pd
import matplotlib.pyplot as plt
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
file1 = open('nonredundantRNA.txt', 'r')
Lines = file1.readlines()
file1.close()
pos_list=[]
neg_list=[]
data_list=[]
for line in Lines:
	pos_list.extend(glob.glob("/home/kihara/modyd/Desktop/RNA_GNN_Nov11/RNA-graph-picklesmorepos/"+line.strip()+"_pos.pkl"))
	neg_list.extend(glob.glob("/home/kihara/modyd/Desktop/RNA_GNN_Nov11/RNA-graph-picklesmorepos/"+line.strip()+"_neg.pkl"))
random.shuffle(neg_list)
random.shuffle(pos_list)
for i in pos_list:
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
print(len(data_list))
for i in neg_list:
	with open(i, 'rb') as f:
	    newdict = pickle.load(f)
	if(len(newdict)==0):
		continue
	for coord,graph in ((newdict.items())):
		graph['key']=coord+i[-12:-8]
		data_list.append(graph)
		#print(graph.keys())
	if(len(data_list)>500000):
		break
print(len(data_list))


parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 250)
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 512)
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
model = utils.initialize_model(model, device)
print('Device: {}'.format(device))
#train and test dataset
random.shuffle(data_list)
train_dataset = data_list[0:int(0.9*len(data_list))]
test_dataset = data_list[int(0.9*len(data_list)):]
num_train_real = len([0 for k in train_dataset if k['C']==1])
num_train_decoy = len([0 for k in train_dataset if k['C']==0])
train_weights = [1/num_train_real if k['C']==1 else 1/num_train_decoy for k in train_dataset]
train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
train_dataloader = DataLoader(train_dataset, args.batch_size, \
     shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn,\
     sampler = train_sampler)
#train_dataloader = DataLoader(train_dataset, args.batch_size, \
    # shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, args.batch_size, \
     shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#loss function
loss_fn = nn.BCELoss()
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
a=[]
b=[]
c=[]
d=[]
e=[]
f=[]
g=[]
h=[]

fig, ax = plt.subplots()
#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

for epoch in range(num_epochs):
    values={}
    st = time.time()
    #collect losses of each iteration
    train_losses = [] 
    test_losses = [] 

    #collect true label of each iteration
    train_true = []
    test_true = []
    
    #collect predicted label of each iteration
    train_pred = []
    test_pred = []
    
    model.train()
    for i_batch, sample in enumerate(train_dataloader):
        model.zero_grad()
        H, A1, A2, Y, V, keys = sample 
        H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device),\
                            Y.to(device), V.to(device)
        
        #train neural network
        pred = model.train_model((H, A1, A2, V))

        loss = loss_fn(pred, Y) 
        loss.backward()
        optimizer.step()
        
        #collect loss, true label and predicted label
        train_losses.append(loss.data.cpu().numpy())
        train_true.append(Y.data.cpu().numpy())
        train_pred.append(pred.data.cpu().numpy())
        #if i_batch>10 : break
    
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
        #if i_batch>10 : break
        
    train_losses = np.mean(np.array(train_losses))
    test_losses = np.mean(np.array(test_losses))
    
    train_pred = np.concatenate(np.array(train_pred), 0)
    test_pred = np.concatenate(np.array(test_pred), 0)
    
    train_true = np.concatenate(np.array(train_true), 0)
    #print(test_true)
    test_true = np.concatenate(np.array(test_true), 0)
    #print(test_true)
    train_roc = roc_auc_score(train_true, train_pred) 
    test_roc = roc_auc_score(test_true, test_pred)
    train_pr = average_precision_score(train_true, train_pred) 
    test_pr = average_precision_score(test_true, test_pred)
    precision, recall, thresholds = precision_recall_curve(test_true, test_pred)
    ax.cla()
    ax.plot(recall, precision, color='purple')
    #display plot
    plt.savefig(str(epoch)+'morepos.png')
    trpred=np.where(train_pred > 0.5, 1,0) 
    tepred=np.where(test_pred > 0.5, 1,0) 
    print(confusion_matrix(train_true, trpred))
    print(confusion_matrix(test_true, tepred))
    print(train_pr,test_pr)
    end = time.time()
    print ("%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" \
    %(epoch, train_losses, test_losses, train_roc, test_roc, end-st))
    a.append(confusion_matrix(train_true, trpred))
    b.append(confusion_matrix(test_true, tepred))
    c.append(train_losses)
    d.append(test_losses)
    e.append(train_roc)
    f.append(test_roc)
    g.append(train_pr)
    h.append(test_pr)
    values['cmtr']=a
    values['cmte']=b
    values['train_losses']=c
    values['test_losses']=d
    values['train_roc']=e
    values['test_roc']=f
    values['train_pr']=g
    values['test_pr']=h
    df = pd.DataFrame(data=values)
    df.to_csv('preds_RNA_DTImorepos_loss2' + '.csv', sep=',', index=False)
    name = save_dir + '/save_'+str(epoch)+'.pt'
    torch.save(model.state_dict(), name)
