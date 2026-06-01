# -*- coding: utf-8 -*-
"""Dataset creation: "more positives" variant.

This variant labels the 8 grid points surrounding each ion (the corners of the
enclosing 3-Angstrom voxel) as positive, rather than a single grid point. It
builds receptor/ligand graph samples from RNA-only PDB files and pickles them
into per-structure dictionaries (all, positive, and negative samples).
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
        #print(atom.GetIdx(),atom.GetSymbol(),atom.GetMonomerInfo().GetName(),atom.GetPDBResidueInfo().GetResidueName(),dm[atom.GetIdx()])
        if(atom.GetSymbol() == ionname):
          finallist.append(dm[atom.GetIdx()])
    return finallist
pdbfiles=(glob.glob("RNA-only-PDB/*.pdb"))
ionname='Mg'

for pdbfile in pdbfiles:
	pdbname=pdbfile[13:17]
	
	full_mol = MolFromPDBFile('RNA-only-PDB/'+pdbname+".pdb", sanitize=False)
	ct = full_mol.GetConformers()[0]
	dt = np.array(ct.GetPositions())
	if(dt.shape[0]>40000):
		print(pdbname+" too large")
		continue
	#try:
		#os.mkdir('RNA-graph-pickles/'+pdbname) 
	#except OSError as error:
		#print(error)
		#if(len(os.listdir('RNA-graph-pickles/'+pdbname))!=0):
			#continue  
			
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

	#print(m)
	#print(d1[54])

	print(receptor_count)

	newlist=[]
	for i in range(int(round_to_3(min(d1[:,0]))),int(max(d1[:,0]))+1,3):
	  for j in range(int(round_to_3(min(d1[:,1]))),int(max(d1[:,1]))+1,3):
	    for k in range(int(round_to_3(min(d1[:,2]))),int(max(d1[:,2]))+1,3):
		    newlist.append([i,j,k])
	newlist=np.array(newlist)
	print(newlist.shape)

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
	  #print(finallist,i)
	  if((finallist.size!=0) and any(np.equal(finallist,i).all(1))):
		  correct=1
		#print(i,finallist)
		  print(i,correct)
	  dm = distance_matrix(d1, d2+i)
	  #dm=np.full((receptor_count,1),10)
	  #print(i)
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
	  #print(agg_adj2.shape,H.shape)
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
	with open('RNA-graph-picklesmorepos/'+pdbname+'.pkl', 'wb') as file:	  

	  pickle.dump(pdbdict, file)
	with open('RNA-graph-picklesmorepos/'+pdbname+'_pos.pkl', 'wb') as file:	  

	  pickle.dump(pdbdictpos, file)
	with open('RNA-graph-picklesmorepos/'+pdbname+'_neg.pkl', 'wb') as file:	  

	  pickle.dump(pdbdictneg, file)

