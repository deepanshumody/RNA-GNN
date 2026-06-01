"""Exploratory / reference script.

Final preprocessing step that selects the best-resolution structure per
sequence-similarity cluster, producing ``nonredundantRNA.txt``. For each
cluster (a comma-separated line in ``OnlyRNAlist.txt``) it parses every
candidate PDB structure, picks the one with the lowest (best) resolution,
and writes its PDB id to the output if the resolution is acceptable.

This is an exploratory data-preparation script kept for reference; it is not
part of the importable package API.
"""

from Bio.PDB import *
count=0
with open("OnlyRNAlist.txt", "r") as inputFile,open("nonredundantRNA.txt","w") as outFile:
	for line in inputFile:
		fin=line.strip().split(',')
		fin.pop()
		
		bestres=10000
		pdbid='a'
		for i in fin:
			parser = PDBParser(QUIET=True)
			structure = parser.get_structure("test", 'RNA-only-PDB/'+i+".pdb")
			resolution = structure.header["resolution"]
			if resolution == None:
				print("nores")
				count+=1
				continue
			if resolution<bestres:
				bestres=resolution
				pdbid=i
		if(bestres==10000):
			print(bestres,pdbid)
		elif(bestres<6):
			print("else",bestres)
			outFile.write(pdbid+"\n")
print(count)
