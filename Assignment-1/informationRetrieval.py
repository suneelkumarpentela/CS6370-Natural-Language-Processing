from util import *
import numpy as np
import math
# Add your import statements here




class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = {}

		#contains all the distinct terms present in all docs
		terms = []
		#dict with terms as keys with their IDF values
		IDF = {}

		for i in range(len(docs)):
			for sentence in docs[i]:
				for token in sentence:
					word = token.lower()
					if word not in terms:
						terms.append(word)
						IDF[word] = 0
					# DF is <=1, so adding 2 is used to filter if 
					# word is present in a particular document. 
					IDF[word] += 2 

			for token in IDF.keys():
				word = token.lower()
				if (IDF[word] > 1):
					IDF[word] += IDF[word] - math.floor(IDF[word]) + 1.0/(len(docs))

		dim = len(terms)

		index["terms"] = terms
		index["dim"] = dim
		index["docIds"] = docIDs

		zero_vector = [0 for i in range(dim)]

		#Each list in TF is a vector representing a doc in terms space.

		#TF = [vector for i in range(len(docs)) ]
		TF = {}
		for docID in docIDs:
			TF[docID] = zero_vector	

		for docID,doc in zip(docIDs,docs):
			for sentence in doc:
				for token in sentence:
					word = token.lower()
					idx = terms.index(word)
					TF[docID][idx] += 1

		# for i in range(len(docs)):
		# 	for sentence in docs[i]:
		# 		for token in sentence:
		# 			word = token.lower()
		# 			TF[i][terms[word]] += 1 

		inv_index = TF

		for docID in docIDs:
			for idx,word in enumerate(terms): 

				inv_index[docID][idx] *= np.log(1.0/IDF[word])

		index["inv_index"] = inv_index
		
		self.index = index


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		terms = self.index["terms"]
		dim = self.index["dim"]
		doc_IDs = self.index["docIDs"]
		inv_index = self.index["inv_index"]

		zero_vector = [0 for i in range(dim)]

		query_vectors = [zero_vector for i in range(len(queries))]

		#converting all the queries to vectors in terms space by means of TF
		for i,query in enumerate(queries):			
			for sentence in query:
				for token in sentence:
					word = token.lower()
					if word in terms:
						idx = terms.index[word]
						query_vectors[i][idx] += 1

		#computing cosine similarity for all queries with docs

		for i,query in enumerate(queries):
			query_vector = np.array(query_vectors[i])  
			cos_sim_dict = {}
			for doc_ID in doc_IDs:
				doc_vector = np.array(inv_index[doc_ID])
				#since comparision is between docs with same query, I skipped norm of query
				cos_sim = np.dot(query_vector,doc_vector)/(np.linalg.norm(doc_vector))
				cos_sim_dict[doc_ID] = cos_sim
			
			doc_ID_ordered = []
			for docID,_ in sorted(cos_sim_dict.items(),key = lambda item : item[1]):
				doc_ID_ordered.append(int(docID))
			doc_IDs_ordered.append(doc_ID_ordered)

		return doc_IDs_ordered