# Add your import statements here
import numpy as np
import math
import pandas as pd
import json
from sklearn.decomposition import TruncatedSVD


# Add any utility functions here
def ground_truth_metrics(query_rels,query_ids):
    query_count = len(query_ids)
    query_ids = sorted(query_ids,key = lambda x : int(x))
    ground_truth_list = [ [] for i in range(query_count)]

    for query_info in query_rels:
        query_val = int(query_info["query_num"])
        if query_val in query_ids:
            query_i = query_ids.index(query_val)
            ground_truth_list[query_i].append(int(query_info["id"]))

    return ground_truth_list


def ground_truth_relevance_metrics(query_rels,query_ids):
    query_count = len(query_ids)
    query_ids = sorted(query_ids,key = lambda x : int(x))
    ground_truth_list = [ [ [],[] ] for i in range(query_count)]

    for query_info in query_rels:
        query_val = int(query_info["query_num"])
        if query_val in query_ids:
            query_i = query_ids.index(query_val)
            ground_truth_list[query_i][0].append( int(query_info["id"]) )
            ground_truth_list[query_i][1].append( int(query_info["position"]) )

    return ground_truth_list

def spell_check(terms,word):
    #selecting only candidates with first letter matching
    candidates = [term for term in terms if term[0]==word[0]]
    corrected_word = closest_candidate(candidates,word)
    return terms.index(corrected_word)

def closest_candidate(candidates,word):
	min_dist = 50
	for candidate in candidates:
		edit_distance = EditDistDP(candidate, word)
	if min_dist > edit_distance:
		min_dist = edit_distance
		res = candidate
	return res

def EditDistDP(str1, str2):
	
	len1 = len(str1)
	len2 = len(str2)

	# Create a DP array to memoize result
	# of previous computations
	DP = [[0 for i in range(len1 + 1)]
			for j in range(2)]

	# Base condition when second String
	# is empty then we remove all characters
	for i in range(0, len1 + 1):
		DP[0][i] = i

	# Start filling the DP
	# This loop run for every
	# character in second String
	for i in range(1, len2 + 1):
		
		# This loop compares the char from
		# second String with first String
		# characters
		for j in range(0, len1 + 1):

			# If first String is empty then
			# we have to perform add character
			# operation to get second String
			if (j == 0):
				DP[i % 2][j] = i

			# If character from both String
			# is same then we do not perform any
			# operation . here i % 2 is for bound
			# the row number.
			elif(str1[j - 1] == str2[i-1]):
				DP[i % 2][j] = DP[(i - 1) % 2][j - 1]
			
			# If character from both String is
			# not same then we take the minimum
			# from three specified operation
			else:
				DP[i % 2][j] = (1 + min(DP[(i - 1) % 2][j],
									min(DP[i % 2][j - 1],
								DP[(i - 1) % 2][j - 1])))
			
	# After complete fill the DP array
	# if the len2 is even then we end
	# up in the 0th row else we end up
	# in the 1th row so we take len2 % 2
	# to get row
	return DP[len2 % 2][len1]

def LSA(inv_index,k=600):
	term_doc_matrix = np.array(list(inv_index.values()))

	svd_model = TruncatedSVD(n_components=k, algorithm='randomized', n_iter=50, random_state=122)
	X_topics = svd_model.fit_transform(term_doc_matrix)

	U = X_topics/ svd_model.singular_values_
	Sigma_matrix = np.diag(svd_model.singular_values_)
	VT = svd_model.components_

	b = np.dot(np.dot(U,Sigma_matrix),VT)
	return b



