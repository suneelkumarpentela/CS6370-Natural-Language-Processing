
from util import *
import numpy as np
import math
import pandas as pd
import json
from sklearn.decomposition import TruncatedSVD

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
		DF = {}

		for doc in docs:
			for sentence in doc:
				for word in sentence:
					if word not in terms:
						terms.append(word)
						DF[word] = 1
						IDF[word] = 0 
					else:
						DF[word] += 1

			for word in DF.keys():
				if(DF[word] > 0):
					IDF[word] += 1
					DF[word] = 0		

		dim = len(terms)	

		#Each list in TF is a vector representing a doc in terms space.

		TF = {}
		inv_index = {}

		for docID in docIDs:
			TF[docID] = [0 for i in range(dim)]
			inv_index[docID] = [0 for i in range(dim)]

		for docID,doc in zip(docIDs,docs):
			for sentence in doc:
				for word in sentence:
					idx = terms.index(word)
					TF[docID][idx] += 1

		for word,_ in IDF.items():
			IDF[word] = np.log( len(docIDs)/IDF[word] )

		for docID in docIDs:
			idx=0
			for word in terms: 
				if (inv_index[docID][idx]==0) :
					inv_index[docID][idx] = TF[docID][idx]*IDF[word]
				idx += 1

		# pd.DataFrame(inv_index).to_csv("inv.csv",index=False)
		# pd.DataFrame(IDF).to_csv("idf.csv",index=False)
		json.dump(inv_index, open("output/"+ "inv_index.json", 'w'))
		json.dump(IDF, open("output/" + "idf.json", 'w'))
		json.dump(terms, open("output/" + "basis_terms.txt", 'w'))
		json.dump(docIDs, open("output/" + "doc_IDs.txt", 'w'))

		'''
		# ## pd.DataFrame(inv_index).to_csv("inv.csv",index=False)
		# ## pd.DataFrame(IDF).to_csv("idf.csv",index=False)
		# json.dump(inv_index, open("output/"+ "inv_index.json", 'w'))
		# json.dump(IDF, open("output/" + "idf.json", 'w'))
		# json.dump(terms, open("output/" + "basis_terms.txt", 'w'))
		# json.dump(docIDs, open("output/" + "doc_IDs.txt", 'w'))

		inv_index = json.load(open("output/"+ "inv_index.json", 'r'))
		IDF = json.load(open("output/"+ "idf.json", 'r'))
		terms = json.load(open("output/"+ "basis_terms.txt", 'r'))
		docIDs = json.load(open("output/"+ "doc_IDs.txt", 'r'))

		#creating the 1400*6600 document-term matrix with tf-idf values.
		# string keys to int keys
		inv_index = dict(zip(list(map(int,inv_index.keys())),inv_index.values()))
		term_doc_matrix = np.array(list(inv_index.values()))

		k = 700

		a = LSA(term_doc_matrix,k=1000)
		

		threshold = 0.7	
		R = np.linalg.norm(term_doc_matrix-a, 'fro')/np.linalg.norm(term_doc_matrix, 'fro')
		print(f'R = {R}')

		############################
		'''

		index["terms"] = terms
		index["docIDs"] = docIDs

		index["IDF"] = IDF
		index["inv_index"] = inv_index

		print(dim)
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
		dim = len(terms)
		doc_IDs = self.index["docIDs"]
		doc_IDs = [int(doc_ID) for doc_ID in doc_IDs]
		IDF = self.index["IDF"]
		inv_index = self.index["inv_index"]
		

		#print(inv_index["2"])

		#zero_vector = [0 for i in range(dim)]

		query_vectors = [[0 for i in range(dim)] for i in range(len(queries))]

		#converting all the queries to vectors in terms space by means of TF
		for i,query in enumerate(queries):			
			for sentence in query:
				for word in sentence:
					if word in terms:
						idx = terms.index(word)
						query_vectors[i][idx] += 1
					else :
						idx = spell_check(terms,word)
						query_vectors[i][idx] += 1

		#multiplying query_vectors with IDF of the terms in it

		for query_vector in query_vectors:
			for i,word in enumerate(terms):
				query_vector[i] = query_vector[i]*IDF[word]


		#computing cosine similarity for all queries with docs

		for i,query in enumerate(queries):
			query_vector = np.array(query_vectors[i])  
			cos_sim_dict = {}

			for doc_ID in doc_IDs:
				doc_vector = np.array(inv_index[doc_ID])

				if np.sum(doc_vector)==0 :
					cos_sim = 0
				else:
					cos_sim = np.dot(query_vector,doc_vector)/(np.linalg.norm(doc_vector)*np.linalg.norm(query_vector))

				cos_sim_dict[doc_ID] = cos_sim

			doc_ID_ordered = []

			for docID,_ in sorted(cos_sim_dict.items(),key = lambda item : item[1],reverse = True):
				doc_ID_ordered.append(int(docID))

			doc_IDs_ordered.append(doc_ID_ordered)

		return doc_IDs_ordered

