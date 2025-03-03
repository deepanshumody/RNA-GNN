import pickle
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import numpy as np
from multiprocessing import Pool
def calcu_drna_seqsim_penalmax(input_seq_list1,input_seq_list2):
    max_similarity1=0
    for input_seq1 in input_seq_list1:
        #print(input_seq1)
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
            #print("denominator %d, match %d, current identity %.4f"%(total_denominator,max_match,cur_sim))
    return max_similarity1
with open('saved_dictionary.pkl', 'rb') as f:
    newdict2 = pickle.load(f)
fin=[]
seq_record = np.ones([len(newdict2), len(newdict2)])
for i in range(len(newdict2) - 1):
	pdb1 = list(newdict2.keys())[i]
	input_seq_list1 = list(newdict2.values())[i]
	p = Pool(48)
	calcu_list = []
	for j in range(i, len(newdict2)):
	    pdb2 = list(newdict2.keys())[j]
	    input_seq_list2 = list(newdict2.values())[j]
	    res = p.apply_async(calcu_drna_seqsim_penalmax, args=(input_seq_list1,input_seq_list2))
	    calcu_list.append(res)
	p.close()
	p.join()
	count_result = 0
	for j in range(i, len(newdict2)):
	    pdb2 = list(newdict2.keys())[j]
	    res = calcu_list[count_result]
	    count_result += 1
	    max_similarity1=res.get()
	    seq_record[i,j]=max_similarity1
	    seq_record[j,i]=max_similarity1#max_similarity2
	    #print("%s/%s: %.4f/%.4f"%(pdb1,pdb2,seq_record[i,j],seq_record[j,i]))
	print("%d/%d calculation finished" % (i, len(newdict2)))
	np.save("2d_matrix.npy", seq_record)
