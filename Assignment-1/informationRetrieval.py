#from util import *
import numpy as np
import math
import pandas as pd
from sklearn.decomposition import TruncatedSVD
#import umap


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

		for i in range(len(docs)):
			for sentence in docs[i]:
				for token in sentence:
					word = token.lower()
					if word not in terms:
						terms.append(word)
						DF[word] = 0
						IDF[word] = 0 

		# print(f'len(terms) = {len(terms)}')
		# print(f'len(docs) = {len(docs)}')

		for doc in docs:
			for sentence in doc:
				for token in sentence:
					word = token.lower()
					#if word in IDF.keys():
					DF[word] += 1

			for word in DF.keys():
				if(DF[word] > 0):
					IDF[word] += 1
					DF[word] = 0			

		dim = len(terms)

		index["terms"] = terms
		index["dim"] = dim
		index["docIDs"] = docIDs

		#print(f'index = {index}')

		#Each list in TF is a vector representing a doc in terms space.

		TF = {}
		inv_index = {}

		for docID in docIDs:
			TF[docID] = [0 for i in range(dim)]
			inv_index[docID] = [0 for i in range(dim)]

		for docID,doc in zip(docIDs,docs):
			for sentence in doc:
				for token in sentence:
					word = token.lower()
					idx = terms.index(word)
					TF[docID][idx] += 1

		for word,_ in IDF.items():
			IDF[word] = np.log( len(docIDs)/IDF[word] )

		for docID in docIDs:
			for idx,word in enumerate(terms): 
				if (inv_index[docID][idx]==0) :
					inv_index[docID][idx] = TF[docID][idx]*IDF[word]

		index["IDF"] = IDF
		index["inv_index"] = inv_index
		
		print(f'(len(list(inv_index.values())) = {len(list(inv_index.values()))}')
		print(f'len(list(inv_index.values())[0]) = {len(list(inv_index.values())[0])}')
		
		#creating the 1400*6600 document-term matrix with tf-idf values. 
		term_doc_matrix = np.array(list(inv_index.values()))
		print(f'Shape of term_doc_matrix = {term_doc_matrix.shape}')
		
		k=200 #try different k values

		svd_model = TruncatedSVD(n_components=k, algorithm='randomized', n_iter=50, random_state=122)
		X_topics = svd_model.fit_transform(term_doc_matrix)

		# for i,comp in enumerate(svd_model.components_):
		# 	terms_comp = zip(terms,comp)
		# 	sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
		# 	print("Topic "+str(i)+": ")
		# 	for t in sorted_terms:
		# 		print(t[0])
		# 		print(" ")

		U = X_topics/ svd_model.singular_values_
		print(f'Shape of U = {U.shape}')
		Sigma_matrix = np.diag(svd_model.singular_values_)
		print(f'Shape of Sigma_matrix = {Sigma_matrix.shape}')
		VT = svd_model.components_
		print(f'Shape of VT = {VT.shape}')

		B= np.dot(np.dot(U,Sigma_matrix),VT)
		print(f'Shape of B = {B.shape}')


		#u,sigma,vt = np. linalg.svd(self.matrix)
		#reconstructedMatrix= dot(dot(u,linalg.diagsvd(sigma,len(self.matrix),len(vt))),vt)

		threshold = 0.7	
		R = np.linalg.norm(term_doc_matrix-B, 'fro')/np.linalg.norm(term_doc_matrix, 'fro')
		print(f'R = {R}')
		

		self.index = index

	# def lsa(self, inv_index):
	#     term_doc_matrix = np.array(list(inv_index.values()))
	#     print(f'Shape of term_doc_matrix = {term_doc_matrix.shape}')

	# 	# SVD represent documents and terms in vectors 
	#     svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
	#     X_topics = svd_model.fit(term_doc_matrix)

	    # terms = vectorizer.get_feature_names()

	    # for i,comp in enumerate(svd_model.components_):
    	#     terms_comp = zip(terms,comp)
    	#     sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    	#     print("Topic "+str(i)+": ")
    	#     for t in sorted_terms:
        # 	    print(t[0])
        # 	    print(" ")

	    # return X_topics
	#     embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)

	#     plt.figure(figsize=(7,5))
	#     plt.scatter(embedding[:, 0], embedding[:, 1], c = dataset.target,
	#     s = 10, # size
	#     edgecolor='none'
	#     )
	#     plt.show()
		


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
		IDF = self.index["IDF"]
		inv_index = self.index["inv_index"]

		#zero_vector = [0 for i in range(dim)]

		query_vectors = [[0 for i in range(dim)] for i in range(len(queries))]

		#converting all the queries to vectors in terms space by means of TF
		for i,query in enumerate(queries):			
			for sentence in query:
				for token in sentence:
					word = token.lower()
					if word in terms:
						idx = terms.index(word)
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