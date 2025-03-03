# RNA-metal-ion-GNN
The code contains 2 Graph Neural Network model architectures for metal ion binding site prediction in RNA-
1. GCN
2. GNN-DTI based on the paper - https://pubs.acs.org/doi/10.1021/acs.jcim.9b00387

Following are the steps required to run it-

## 1. Obtaining nonredundantRNA.txt
To reproduce everything including which molecules to select for training, *clustering1.py, clustering2.py, clustering3.py, bestresolutionfromcluster.py* need to be run in the given order.

This would require an input of comma separated list of all RNA only PDB IDs which is provided in *OnlyRNAlist.txt*. This list can also be obtained from RCSB PDB website by searching for RNA only structures.

Running the above will output a file *nonredundantRNA.txt* which contains a list of non redundant RNAs based on sequence having resolution better than 6A.

These files have already been run and *nonredundantRNA.txt* contains a list of PDB IDs to train on but the above procedure can be followed to reproduce it. Hence, it is optional.

## 2. Creating the dataset
Currently, a few versions of the dataset have been created. PDB structures of the RNA are required to be downloaded and saved in folder RNA-only-PDB in the working directory. The dataset will be created from these PDB files.

Then *gnn_rna.py, gnn_rna_0A.py, gnn_rna_autodock.py, gnn_rna_morepos.py* can be used to create different versions of the dataset by choosing different locations to place the ions. For each PDB, a pickle file is generated containing all locations and the generated graphs and features. The GNN would work as a binary classifier for each location.

*MG_ideal.sdf* is the file for the binding ion.

*gnn_rna_autodock.py* requires additional files generated through autodock vina

## 2.1 Using the datasets created
The working directory is /home/kihara/modyd/Desktop/RNA_GNN_Metal_ion

There are 4 folders in the working directory with 4 datasets present in the folders named-

1.RNA-graph-pickles - Contains points placed on a 3A grid over the whole molecule, the closest point to the actual location of ion being considered positive. Contains roughly 3000 positive points and 1000000 negative points.

2.RNA-graph-pickles0A - Contains points placed on a 3A grid over the whole molecule but stops at 0A from the atoms at the edges, the closest point to the actual location of ion being considered positive. Contains roughly 2500 positive points and 500000 negative points.

3.RNA-graph-pickles-autodock - Contains points placed over the whole molecule using Autodock Vina.

4.RNA-graph-picklesmorepos - Contains points placed on a 3A grid over the whole molecule, the 8 closest cubic grid points to the actual location of ion being considered positive. Contains roughly 12000 positive points and 1000000 negative points.

The creation of training, validation and testing sets is done by the training code itself. The commands following in the section below can be run to train the models.

## 3.a) Training GCN
*train_gnn.py* is the main version of the GCN. It can be used to train on any version of the dataset by changing the input folder location in the code. The best model is saved and *predfrommodel.py* can be used to test individual PDB structures.

cd to /home/kihara/modyd/Desktop/RNA_GNN_Metal_ion

command - python train_gnn.py

## 3.b) Training GNN-DTI
*train.py* inside model_GNN_DTI folder can be run. The dataset needs to be outside this folder. It can be used to train on any version of the dataset by changing the input folder location in the code. All of the models are saved and can be tested on individual PDBs using *test.py*

cd to /home/kihara/modyd/Desktop/RNA_GNN_Metal_ion/model_GNN_DTI

command - CUDA_VISIBLE_DEVICES="x" python train.py where x is the GPU #


## 4.a) Testing individual structures GCN
cd to /home/kihara/modyd/Desktop/RNA_GNN_Metal_ion

command - python predfrommodel.py

## 4.b) Testing individual structures GNN-DTI
cd to /home/kihara/modyd/Desktop/RNA_GNN_Metal_ion/model_GNN_DTI

command - CUDA_VISIBLE_DEVICES="x" python test.py where x is the GPU #



## Links to slides-

https://purdue0-my.sharepoint.com/:p:/g/personal/modyd_purdue_edu/EScfuI-3dBpDghQkYHrS6P8BsGjR4hLAvLPKfm-EYOnFNg?e=tvIA8d

https://purdue0-my.sharepoint.com/:p:/g/personal/modyd_purdue_edu/EXJN6pxMfNZBnivvTjUdbCABFum4tNid0VJ6X5CW7WLyXA?e=6eIJPs

https://purdue0-my.sharepoint.com/:p:/g/personal/modyd_purdue_edu/EdZh7vnDzwZClZ6i372E_DUB3SrtWZm17wQpZd03VlAa8w?e=kDzZlA
