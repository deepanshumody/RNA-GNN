# -*- coding: utf-8 -*-
"""Dataset creation for the RNA-GNN model (AutoDock Vina variant).

This variant places candidate ion sites using AutoDock Vina docking output
(read from the ``Mg_RNA_Docked`` directory) instead of grid-only sampling.
For each non-redundant RNA structure it builds receptor/ligand atom-feature
graphs and serializes them as pickles for downstream GNN training.
"""

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolfiles import MolFromPDBFile
import numpy as np
from scipy.spatial import distance_matrix
import pickle
import os
import glob
import shelve

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Shared atom featuriser — single source of truth; fixes the Mg->H encoding bug.
from atom_features import get_atom_feature

def round_to_3(number):
    return 3 * round(number / 3)

def mol_with_atom_index(mol,dm):
    finallist=[]
    for atom in mol.GetAtoms():
        if(atom.GetSymbol() == ionname):
          finallist.append(dm[atom.GetIdx()])
    return finallist
pdbfiles=(glob.glob("RNA-only-PDB/*.pdb"))
ionname='Mg'

file1 = open('nonredundantRNA.txt', 'r')
Lines = file1.readlines()
file1.close()
for line in Lines:
	pdbname=line.strip()
	
	full_mol = MolFromPDBFile('RNA-only-PDB/'+pdbname+".pdb", sanitize=False)
	ct = full_mol.GetConformers()[0]
	dt = np.array(ct.GetPositions())
	if(dt.shape[0]>40000):
		print(pdbname+" too large")
		continue

	with open('RNA-only-PDB/'+pdbname+".pdb", "r") as inputFile,open('RNA-only-PDB-clean/'+pdbname+"_clean.pdb","w") as outFile:
	   for line in inputFile:
	       if not line.startswith("HETATM"):
		       outFile.write(line)
	

	pro_mol = MolFromPDBFile('RNA-only-PDB-clean/'+pdbname+"_clean.pdb", sanitize=False)
	receptor_count = pro_mol.GetNumAtoms()
	c=pro_mol.GetConformers()
	if len(c)==0:
		continue
	c1 = c[0]
	d1 = np.array(c1.GetPositions())
	adj1 = GetAdjacencyMatrix(pro_mol)+np.eye(receptor_count)

	


	mol_with_atom_index(pro_mol,d1)

	finallist=mol_with_atom_index(full_mol,dt)

	cor=[]
	for i in finallist:
	  cor.append([3*np.floor(i[0]/3),3*np.floor(i[1]/3),3*np.floor(i[2]/3)])
	  cor.append([3*np.floor(i[0]/3),3*np.floor(i[1]/3),3*np.ceil(i[2]/3)])
	  cor.append([3*np.floor(i[0]/3),3*np.ceil(i[1]/3),3*np.floor(i[2]/3)])
	  cor.append([3*np.ceil(i[0]/3),3*np.floor(i[1]/3),3*np.floor(i[2]/3)])
	  cor.append([3*np.ceil(i[0]/3),3*np.ceil(i[1]/3),3*np.floor(i[2]/3)])
	  cor.append([3*np.ceil(i[0]/3),3*np.floor(i[1]/3),3*np.ceil(i[2]/3)])
	  cor.append([3*np.floor(i[0]/3),3*np.ceil(i[1]/3),3*np.ceil(i[2]/3)])
	  cor.append([3*np.ceil(i[0]/3),3*np.ceil(i[1]/3),3*np.ceil(i[2]/3)])
	finallist=np.array(cor)


	m = Chem.rdmolfiles.MolFromMolFile(ionname.upper()+'_ideal.sdf')

	print(receptor_count)

	newlist=[]
	with open("./Mg_RNA_Docked/"+pdbname.lower()+"_mg_out.pdb", "r") as inputFile:
	  for line in inputFile:
	    if line.startswith("HETATM"):
	      newlist.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
	newlist=np.array(newlist)
	newlist=3*(np.around(np.array(newlist)/3))

	print(adj1.shape)

	receptor_feature = get_atom_feature(pro_mol, is_ligand=False)

	print((receptor_feature.shape))

	ligand_count = m.GetNumAtoms()
	ligand_feature = get_atom_feature(m, is_ligand=True)

	c2 = m.GetConformers()[0]
	d2 = np.array(c2.GetPositions())
	adj2 = GetAdjacencyMatrix(m) + np.eye(ligand_count)
	
	pdbdict={}
	pdbdictpos={}
	pdbdictneg={}
	for i in newlist:
	  correct=0
	  if((finallist.size!=0) and any(np.equal(finallist,i).all(1))):
		  correct=1
		  print(i,correct)
	  dm = distance_matrix(d1, d2+i)
	  bool_mask=(dm<8).reshape(receptor_count)
	  finaldm=dm[bool_mask]
	  if(finaldm.shape==(0,1)):
		  continue
	  finalrf=receptor_feature[bool_mask]
	  finalrf.shape[0]
	  finaladj1=adj1[bool_mask,:][:,bool_mask]
	  H = np.concatenate([finalrf, ligand_feature], 0)
	  agg_adj1 = np.zeros((finalrf.shape[0] + ligand_count, finalrf.shape[0] + ligand_count))
	  agg_adj1[:finalrf.shape[0], :finalrf.shape[0]] = finaladj1
	  agg_adj1[finalrf.shape[0]:, finalrf.shape[0]:] = adj2  # array without r-l interaction
	  agg_adj2 = np.copy(agg_adj1)
	  agg_adj2[:finalrf.shape[0], finalrf.shape[0]:] = np.copy(finaldm)
	  agg_adj2[finalrf.shape[0]:, :finalrf.shape[0]] = np.copy(np.transpose(finaldm))
	  valid = np.zeros((finalrf.shape[0] + ligand_count,))
	  valid[:receptor_count] = 1
	  sample = {
		 'H': H,
		 'A1': agg_adj1,
		 'A2': agg_adj2,
		 'V': valid,
		 'C': correct
		 }
	  pdbdict[str(i)]=sample
	  if sample['C']==1:
	  	pdbdictpos[str(i)]=sample
	  else:
	  	pdbdictneg[str(i)]=sample
	with open('RNA-graph-pickles-autodock/'+pdbname+'.pkl', 'wb') as file:	  

	  pickle.dump(pdbdict, file)
	with open('RNA-graph-pickles-autodock/'+pdbname+'_pos.pkl', 'wb') as file:	  

	  pickle.dump(pdbdictpos, file)
	with open('RNA-graph-pickles-autodock/'+pdbname+'_neg.pkl', 'wb') as file:	  

	  pickle.dump(pdbdictneg, file)
