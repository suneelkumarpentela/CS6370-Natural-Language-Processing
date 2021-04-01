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
					IDF[word] += TDF[word] - math.floor(IDF[word]) + 1.0/(len(docs))

		index["terms"] = terms
		index["docIds"] = docIDs

		dim = len(terms)
		zero_vector = [0 for i in range(dim)]

		#Each list in TF is a vector representing a doc in terms space.
		
		TF = {}
		for docID in docIDs:
			TF[docID] = zero_vector	

		for docID,doc in zip(docIDs,docs):
			for sentence in doc:
				for token in sentence:
					word = token.lower()
					idx = terms.index(word)
					TF[docID][idx] += 1

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

		#Fill in code here
	
		return doc_IDs_ordered




