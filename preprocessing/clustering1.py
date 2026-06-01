"""Exploratory / reference preprocessing script (step 1 of building the
non-redundant RNA list).

This module performs sequence-similarity clustering of RNA PDB entries: it
reads a list of RNA PDB IDs, groups their associated sequence files, and
computes pairwise sequence-identity to support redundancy filtering.

NOTE: This is an exploratory notebook-style dump kept for reference. It is not
guaranteed to run as-is and may require adaptation (e.g. defining input file
paths and the `files`/`lenh` variables, and resolving imports) before use.
"""

with open('OnlyRNAlist.txt','r') as file:
        pdb_id_list = file.readline().split(',')
        print(pdb_id_list)
pdb_id_list[-1]=pdb_id_list[-1].strip()
newdict={}
for i in pdb_id_list:
  for j in files:
    if(i == j[lenh:lenh+4]):
      if i in newdict:
        (newdict[i].append(j))
      else:
        newdict[i]=[j]
def calcu_drna_seqsim_penalmax(input_seq_list1,input_seq_list2):
    max_similarity1=0
    for input_seq1 in input_seq_list1:
        print(input_seq1)
        if len(input_seq1) <= 10:
            continue
        for input_seq2 in input_seq_list2:
            if len(input_seq2) <= 10:
                continue
            matches = pairwise2.align.globalms(input_seq1,input_seq2,1,-1,-100,-100)
            #may have multiple matches
            max_match=0
            for match_example in matches:
                output_seq = format_alignment(*match_example)
                output_seq_list = output_seq.split("\n")
                match_indicator = output_seq_list[1]
                current_match = 0
                for indicator in match_indicator:
                    if indicator=='|':
                        current_match+=1
                if current_match>=max_match:
                    max_match=current_match
            total_denominator = (len(input_seq1)+len(input_seq2))/2
            cur_sim = max_match/total_denominator
            if cur_sim>max_similarity1:
                max_similarity1 = cur_sim
            print("denominator %d, match %d, current identity %.4f"%(total_denominator,max_match,cur_sim))
    return max_similarity1
matches = pairwise2.align.globalms(input_seq1,input_seq2,1,-1,-100,-100)
def read_seq(input_path1):
    input_seq_list = []
    with open(input_path1,'r') as file:
        line = file.readline()
        while line:
            line = line.strip("\n")
            if(line.startswith('>')):
              line = file.readline()
              continue
            input_seq_list.append(line)
            line = file.readline()
    return input_seq_list
    newdict2={}
for i in pdb_id_list:
  finallist=[]
  for j in newdict[i]:
    input_seq_list = read_seq(j)
    finallist.append(input_seq_list[0])
  newdict2[i]=finallist
import pickle
with open('saved_dictionary.pkl', 'wb') as f:
    pickle.dump(newdict2, f)
