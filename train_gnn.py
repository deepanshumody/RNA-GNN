from sklearn.metrics import RocCurveDisplay
from ogb.graphproppred import Evaluator
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
print(torch.__version__)
import os
import pandas as pd
import torch.nn.functional as F
import glob
import copy
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import random

  # If you use GPU, the device should be cuda
print('Device: {}'.format(device))
pos_pickles=(glob.glob("./RNA-graph-pickles0A/*_pos.pkl"))
neg_pickles=(glob.glob("./RNA-graph-pickles0A/*_neg.pkl"))
random.shuffle(pos_pickles)
random.shuffle(neg_pickles)
#print(pos_pickles)
data_list=[]
#negative_only_test_list=[]
pdb4RUM_list=[]

for i in pos_pickles:
	#savelist=[]
	with open(i, 'rb') as f:
	    newdict2 = pickle.load(f)
	if(len(newdict2)==0):
		continue
	for coord,graph in ((newdict2.items())):
		G=nx.from_numpy_matrix(np.array(graph['A1']))
		for node in G.nodes:
	  		G.nodes[node]['x'] = torch.tensor(graph['H'][node],dtype=torch.int32)
		pyg_graph = from_networkx(G)
		pyg_graph.y=torch.tensor(graph['C'])
		pyg_graph.coords=coord
		pyg_graph.pdb= i[-12:-8]
		data_list.append(pyg_graph)
		#savelist.append(pyg_graph)
	#torch.save(savelist,'./saved_tensors/'+i[-12:-8]+'pos_graph.pt')
	#if(len(data_list)>31):
		#break
pos_samples=len(data_list)
#torch.save(data_list,'./saved_tensors/pos_graphs.pt')
for i in neg_pickles:
	#savelist=[]
	with open(i, 'rb') as f:
	    newdict2 = pickle.load(f)
	if(len(newdict2)==0):
		continue
	for coord,graph in (newdict2.items()):
		G=nx.from_numpy_matrix(np.array(graph['A1']))
		for node in G.nodes:
	  		G.nodes[node]['x'] = torch.tensor(graph['H'][node],dtype=torch.int32)
		pyg_graph = from_networkx(G)
		pyg_graph.y=torch.tensor(graph['C'])
		pyg_graph.coords=coord
		pyg_graph.pdb= i[-12:-8]
		#savelist.append(pyg_graph)
	#torch.save(savelist,'./saved_tensors/'+i[-12:-8]+'neg_graph.pt')
		#if(len(data_list)<2*(pos_samples)):
		data_list.append(pyg_graph)
		#else:
			#negative_only_test_list.append(pyg_graph)
	#print(len(data_list),len(negative_only_test_list))
	#if(len(negative_only_test_list)>=4*(pos_samples)):
		#negative_only_test_list.extend(data_list[0:20])
		#break
	#if(len(data_list)>=2*(pos_samples)):
		#break
with open('./4RUM.pkl', 'rb') as f:
	    newdict2 = pickle.load(f)	
for coord,graph in (newdict2.items()):
		G=nx.from_numpy_matrix(np.array(graph['A1']))
		for node in G.nodes:
	  		G.nodes[node]['x'] = torch.tensor(graph['H'][node],dtype=torch.int32)
		pyg_graph = from_networkx(G)
		pyg_graph.y=torch.tensor(graph['C'])
		pyg_graph.coords=coord
		pyg_graph.pdb= '4RUM'
		pdb4RUM_list.append(pyg_graph)
torch.save(pdb4RUM_list,'pd4rum_graphs.pt')	
  
random.shuffle(data_list)
#print(data_list[0])
print(len(data_list))
#samplist=(random.sample(range(len(data_list)),len(data_list)))
train_loader = DataLoader(data_list[0:int(0.8*len(data_list))], batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(data_list[int(0.8*len(data_list)):int(0.9*len(data_list))], batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(data_list[int(0.9*len(data_list)):], batch_size=32, shuffle=False, num_workers=0)
#negative_only_loader = DataLoader(negative_only_test_list, batch_size=32, shuffle=False, num_workers=0)
pdb4RUM_loader = DataLoader(pdb4RUM_list, batch_size=32, shuffle=True, num_workers=0)
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):


        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)                             
                for i in range(num_layers-2)] + 
            [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]    
        )

        # A list of 1D batch normalization layers
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=hidden_dim) 
                for i in range(num_layers-1)
        ])

        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):

        out = None
 
        for conv, bn in zip(self.convs[:-1], self.bns):
            x1 = F.relu(bn(conv(x, adj_t)))
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1
        x = self.convs[-1](x, adj_t)
        out = x if self.return_embeds else self.softmax(x)

        return out
args = {
      'device': device,
      'num_layers': 5,
      'hidden_dim': 256,
      'dropout': 0.5,
      'lr': 0.001,
      'epochs': 30,
  }

from torch_geometric.nn import global_add_pool, global_mean_pool
#from ogb.graphproppred.mol_encoder import AtomEncoder
full_atom_feature_dims = [2 for i in range(56)]
class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding
### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()
        full_atom_feature_dims = [2 for i in range(56)]
        # Load encoders for Atoms in molecule graphs
        self.node_encoder = AtomEncoder(hidden_dim)

        # Node embedding model
        self.gnn_node = GCN(hidden_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = global_mean_pool

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):


        # Extract important attributes of our mini-batch
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)

        out = None

        embed = self.gnn_node(embed, edge_index)
        features = self.pool(embed, batch)
        out = self.linear(features)


        return out
def train(model, device, data_loader, optimizer, loss_fn):

    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
      batch = batch.to(device)

      if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
          pass
      else:
        is_labeled = batch.y == batch.y

        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out[is_labeled], batch.y[is_labeled].float().unsqueeze(1))

        loss.backward()
        optimizer.step()

    return loss.item()

def eval(model, device, loader, evaluator, save_model_results=False, save_file=None):
    model.eval()
    y_true = []
    y_pred = []
    pdb_id = []
    coord = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
            pdb_id.append(batch.pdb)
            coord=coord+(batch.coords) 
            #print("ok",batch.pdb)
      
    coord=np.array(coord)
    pdb_id=np.array(np.concatenate(pdb_id).flat)
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    #print(y_true.shape)
    #print(pdb_id.shape)
    #print(coord.shape)
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if save_model_results:
        print ("Saving Model Predictions")
        # Create a pandas dataframe with a two columns
        # y_pred | y_true
        data = {}
        data['y_pred'] = y_pred.reshape(-1)
        data['y_true'] = y_true.reshape(-1)
        data['pdb_id'] = pdb_id.reshape(-1)
        data['coord'] = coord.reshape(-1)
        df = pd.DataFrame(data=data)
        # Save to csv
        df.to_csv('preds_RNA' + save_file + '.csv', sep=',', index=False)
        RocCurveDisplay.from_predictions(data['y_true'], data['y_pred'])
        plt.savefig("squares.png")
    return evaluator.eval(input_dict)
model = GCN_Graph(args['hidden_dim'],
              1, args['num_layers'],
              args['dropout']).to(device)
evaluator = Evaluator(name='ogbg-molhiv')
model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = torch.nn.BCEWithLogitsLoss()

best_model = None
best_valid_acc = 0

for epoch in range(1, 1 + args["epochs"]):
	print('Training...')
	loss = train(model, device, train_loader, optimizer, loss_fn)

	print('Evaluating...')
	train_result = eval(model, device, train_loader, evaluator)
	val_result = eval(model, device, valid_loader, evaluator)
	test_result = eval(model, device, test_loader, evaluator)

	train_acc, valid_acc, test_acc = train_result['rocauc'], val_result['rocauc'], test_result['rocauc']
	if valid_acc > best_valid_acc:
		best_valid_acc = valid_acc
		best_model = copy.deepcopy(model)
	print(f'Epoch: {epoch:02d}, '
	  f'Loss: {loss:.4f}, '
	  f'Train: {100 * train_acc:.2f}%, '
	  f'Valid: {100 * valid_acc:.2f}% '
	  f'Test: {100 * test_acc:.2f}%')
train_acc = eval(best_model, device, train_loader, evaluator)['rocauc']
valid_acc = eval(best_model, device, valid_loader, evaluator, save_model_results=True, save_file="valid")['rocauc']
test_acc  = eval(best_model, device, test_loader, evaluator, save_model_results=True, save_file="test")['rocauc']
torch.save(best_model.state_dict(), "./best_model.pth")
#negative_only_acc = eval(best_model, device, negative_only_loader, evaluator, save_model_results=True, save_file="negonly")['rocauc']
pdb4RUM_acc = eval(best_model, device, pdb4RUM_loader, evaluator, save_model_results=True, save_file="4RUM")['rocauc']

print(f'Best model: '

f'Train: {100 * train_acc:.2f}%, '

f'Valid: {100 * valid_acc:.2f}% '

f'Test: {100 * test_acc:.2f}%')
"""

