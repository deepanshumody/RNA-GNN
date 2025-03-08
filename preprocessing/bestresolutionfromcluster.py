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
