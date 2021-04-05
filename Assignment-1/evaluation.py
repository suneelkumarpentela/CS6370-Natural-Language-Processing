from util import *
import numpy as np

# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1
		rel_ret_count = 0

		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				rel_ret_count += 1

		precision = rel_ret_count/k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = 0
		query_count = len(query_ids)
		query_ids = sorted(query_ids)

		#imported from util.py
		ground_truth_list = ground_truth_metrics(qrels,query_ids)

		# ground_truth_list = [ [] for i in range(query_count)]

		# for query_info in qrels:
		# 	query_val = int(query_info["query_num"])
		# 	if query_val in query_ids: 
		# 		query_i = query_ids.index(query_val)
		# 		ground_truth_list[query_i].append(int(query_info["id"]))

		for i,query_id in enumerate(query_ids):
			# if(i==0):
			# 	print(doc_IDs_ordered[i],"\n\n\n",ground_truth_list[i])
			precision = self.queryPrecision(doc_IDs_ordered[i],query_id,ground_truth_list[i],k)
			meanPrecision += precision

		meanPrecision = meanPrecision/query_count

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		rel_ret_count = 0
		rel_count = 1
		if ( len(true_doc_IDs) > 0 ):
			rel_count = len(true_doc_IDs)

		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				rel_ret_count += 1

		recall = rel_ret_count/rel_count

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = 0

		query_count = len(query_ids)

		ground_truth_list = ground_truth_metrics(qrels,query_ids)

		#print(ground_truth_list)

		for i,query_id in enumerate(query_ids):

			recall = self.queryRecall(doc_IDs_ordered[i],query_id,ground_truth_list[i],k)
			meanRecall += recall
		
		meanRecall= meanRecall/query_count

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall  = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		fscore = 0
		if ( (precision+recall) > 0):
			fscore = (2*precision*recall)/(precision+recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = 0

		query_count = len(query_ids)
		ground_truth_list = ground_truth_metrics(qrels,query_ids)

		for i,query_id in enumerate(query_ids):
			Fscore = self.queryFscore(doc_IDs_ordered[i], query_id, ground_truth_list[i], k)
			meanFscore += Fscore

		meanFscore = meanFscore/query_count
		
		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		ground_truth_docs = true_doc_IDs[0]
		ground_truth_relevance = true_doc_IDs[1]

		k_predicted_docs = query_doc_IDs_ordered[:k]
		ideal_rel_list = []

		#Actual DCG@k

		DCG  = 0
		for i,doc_ID in enumerate(k_predicted_docs) :
			if doc_ID in ground_truth_docs:
				idx = ground_truth_docs.index(doc_ID)
				rel = ground_truth_relevance[idx]
				ideal_rel_list.append([rel,i+1])
				DCG += rel/(np.log2(i+2))


		ideal_rel_list = sorted(ideal_rel_list,key = lambda x : x[0])
		#Ideal DCG@k

		IDCG = 0
		for i in range(len(ideal_rel_list)):
			rel = ideal_rel_list[i][0]
			pos = ideal_rel_list[i][1]
			IDCG += ( rel/np.log2(pos+1) )
			
		nDCG = DCG/IDCG
		
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = 0

		query_count = len(query_ids)

		ground_truth_list = ground_truth_relevance_metrics(qrels,query_ids)

		for i,query_id in enumerate(query_ids):
			NDCG = self.queryNDCG(doc_IDs_ordered[i], query_id, ground_truth_list[i], k)
			meanNDCG += NDCG

		meanNDCG = meanNDCG/query_count			

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = 0

		doc_count = 0
		rel_ret_count = 0

		for i in range(k):
			doc_count += 1
			
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				rel_ret_count += 1
				if (doc_count):
					avgPrecision += (rel_ret_count/doc_count)
		if(rel_ret_count):
			avgPrecision = avgPrecision/rel_ret_count
		
		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = 0

		query_count = len(query_ids)

		ground_truth_list = ground_truth_metrics(q_rels,query_ids)

		for i,query_id in enumerate(query_ids):

			avgPrecision = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, ground_truth_list[i], k)
			meanAveragePrecision += avgPrecision

		meanAveragePrecision = meanAveragePrecision/query_count

		return meanAveragePrecision

