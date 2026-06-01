"""Exploratory clustering step 3 (reference only).

This script performs agglomerative clustering on a precomputed 2D distance
matrix and writes the resulting PDB-id clusters to a text file. It is an
exploratory/reference notebook dump and is not part of the production pipeline;
it may rely on undefined names (e.g. ``pdb_id_list``) and is not guaranteed to
run as-is.
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import os
a=np.load('/content/drive/MyDrive/2d_matrix.npy')

clustering = AgglomerativeClustering(n_clusters=None, 
            affinity='precomputed', compute_full_tree='True', linkage='complete', distance_threshold=-0.8)
clustering.fit_predict(a)
predict_labels = clustering.labels_
Cluster_Dict = defaultdict(list)
for k in range(len(pdb_id_list)):
    seq = pdb_id_list[k]
    cluster_id = int(predict_labels[k])
    Cluster_Dict[cluster_id].append(seq)
count=0
with open("b.txt",'w') as file:
    for key in Cluster_Dict:
        current_cluster = Cluster_Dict[key]
        for pdb_id in current_cluster:
            count+=1
            file.write(pdb_id+",")
        file.write("\n")
print("in total %d pdbs are clustered"%count)
