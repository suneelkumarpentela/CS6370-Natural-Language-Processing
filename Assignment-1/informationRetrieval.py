from util import *
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

		index = None

		terms = {}
		IDF = {}
		dim = 0
		for i in range(len(docs)):
			for sentence in docs[i]:
				for token in sentence:
					word = token.lower()
					if word not in terms.keys():
						dim+=1
						terms[word] = dim-1
						IDF[word] = 0
					IDF[word] += 2

			for word in IDF.keys():
				if(IDF[word] > 1):
					IDF[word] += math.floor(IDF[word]) + 1.0/(len(docs))


		basis = [0 for i in range(dim)]
		TF = [basis for i in range(len(docs)) ]				
		for i in range(len(docs)):
			for sentence in docs[i]:
				for token in sentence:
					word = token.lower()
					TF[i][terms[word]] += 1 
		inv_index = TF

		for i in range(len(docs)):
			for word in TF[i].keys():
				inv_index[i][word] *= np.log(len(docs)/IDF[word])

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




