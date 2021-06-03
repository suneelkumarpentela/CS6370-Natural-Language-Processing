from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
import itertools
from util import *

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

import numpy as np
import math
import pandas as pd
from gensim.models import Word2Vec
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from datetime import datetime
import json

parser = argparse.ArgumentParser(description='word2vec.py')

parser.add_argument('-dataset', default = "output/", help = "Path to the dataset folder")

print('Starting at: ', str(datetime.now()))

args = parser.parse_args()
docs_txt = json.load(open(args.dataset + "stopword_removed_docs.txt", 'r'))[:]
queries_txt = json.load(open(args.dataset + "stopword_removed_queries.txt", 'r'))[:]

docs_json = json.load(open("cranfield/cran_docs.json", 'r'))[:]
df_docs = pd.DataFrame.from_dict(docs_json, orient='columns')

df_docs['cleaned']=df_docs['body'].apply(lambda x:x.lower())

def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\S+", "", text)
    text=re.sub('[^a-z]',' ',text)
    return text
 
# Cleaning corpus using RegEx
df_docs['cleaned']=df_docs['cleaned'].apply(lambda x: clean_text(x))
# Removing extra spaces
df_docs['cleaned']=df_docs['cleaned'].apply(lambda x: re.sub(' +',' ',x))
# Stopwords removal & Lemmatizing tokens using SpaCy
nlp = spacy.load('en_core_web_sm',disable=['ner','parser'])
nlp.max_length=5000000
# Removing Stopwords and Lemmatizing words
df_docs['lemmatized']=df_docs['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))

quers_json = json.load(open("cranfield/cran_queries.json", 'r'))[:]
df_quers = pd.DataFrame.from_dict(quers_json, orient='columns')

df_quers['cleaned'] = df_quers['query'].apply(lambda x:x.lower())

# Cleaning queries using RegEx
df_quers['cleaned']=df_quers['cleaned'].apply(lambda x: clean_text(x))

# Removing extra spaces
df_quers['cleaned']=df_quers['cleaned'].apply(lambda x: re.sub(' +',' ',x))

combined_training=pd.concat([df_docs.rename(columns={'lemmatized':'text'})['text'],\
                             df_quers.rename(columns={'cleaned':'text'})['text']])\
                             .sample(frac=1).reset_index(drop=True)
                             
print(f'combined_training')
print(combined_training)

train_data=[]
for i in combined_training:
    train_data.append(i.split())

# Training a word2vec model from the given data set
w2v_model = Word2Vec(train_data, size=50, min_count=1,window=5, sg=1,seed = 1, workers=1)
# print(w2v_model.most_similar('airplane'))
print('Vocabulary size:', len(w2v_model.wv.vocab))

my_dict = {}
for idx, key in enumerate(w2v_model.wv.vocab):
    my_dict[key] = w2v_model.wv[key]

# print(my_dict.keys())
# with open('file.txt','w') as data: 
#       data.write(str(my_dict))
#w2v_model.save("word2vec.model")

def get_embedding_w2v(doc_tokens):
    embeddings = []
    if len(doc_tokens)<1:
        return np.zeros(50)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.vocab:
                embeddings.append(w2v_model.wv.word_vec(tok))
            else:
                embeddings.append(np.random.rand(50))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)

# Getting Word2Vec Vectors for Testing Corpus and Queries
df_docs['vector']=df_docs['lemmatized'].apply(lambda x :get_embedding_w2v(x.split()))
df_quers['vector']=df_quers['cleaned'].apply(lambda x :get_embedding_w2v(x.split()))

print('df_docs: ')
print(df_docs)
print('df_quers: ')
print(df_quers)
"""
def average_precision(qid,qvector):
    result_json = json.load(open("cranfield/cran_qrels.json", 'r'))[:]
    df_result = pd.DataFrame.from_dict(result_json, orient='columns')
    # print('df_result')
    # print(df_result.head())
    
    qresult=df_result.loc[df_result['query_num']==str(qid),['id','position']]
    qresult['id']=qresult['id'].astype(int)
    # print('qresult.head():')
    # print(qresult.head())
    qcorpus=df_docs.loc[df_docs['id'].isin(qresult['id']),['id','vector']]
    qcorpus['id']=qcorpus['id'].astype(int)
    # print('qcorpus.head():')
    # print(qcorpus.head())
    print(f'qid = {qid}, Size of qresult: {qresult.shape}, Size of qcorpus: {qcorpus.shape}')
    
    qresult=pd.merge(qresult,qcorpus,on='id')
    # print('qresult again: ')
    # print(qresult.head())

    # Ranking documents for the query
    qresult['similarity']=qresult['vector'].apply(lambda x: cosine_similarity(np.array(qvector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
    qresult.sort_values(by='similarity',ascending=False,inplace=True)
    print('qresult.head() again:')
    # print(qresult.head())
    print(f'Size of qresult now: {qresult.shape}')
    
    # Taking Top 5 documents for the evaluation
    ranking=qresult.head(qresult.shape[0])['position'].values
    print(f'ranking = {ranking}')

    # Calculating precision
    precision=[]
    for i in range(1,qresult.shape[0]+1):
        if ranking[i-1]:
            # print(f'np.sum(ranking[:i]) {np.sum(ranking[:i])}')
            # print(f'len(range(1,qresult.shape[0]+1) = {len(range(1,qresult.shape[0]+1))}')
            precision.append(np.sum(ranking[:i])/i)
            
    # If no relevant document in list then return 0
    if precision==[]:
        return 0
   
    return np.mean(precision)

def evaluateDataset(self):
		'''
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		'''

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = list(df_quers['cleaned'])

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = list(df_docs['lemmatized'])

		# Build document index
		
		# Rank the documents for each query
        doc_IDs_ordered = []
        for i in range(len(df_quers)):
            doc_IDs_order = list(ranking_ir(df_quers['id'][i]).values())
            doc_IDs_ordered.append(doc_IDs_ordered)

		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
		
		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		
		precisions = []
		
		for k in range(1, 11):
			precision = meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)

		# Plot the metrics and save plot 
		plt.plot(range(1, 11), precisions, label="Mean Precision")

		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.savefig(args.out_folder + "eval_plot_w2v.png")
"""
def ranking_ir(query):
  
  # pre-process Query
  query=query.lower()
  query=clean_text(query)
  query=re.sub(' +',' ',query)

  # generating vector
  vector=get_embedding_w2v(query.split())

  # ranking documents
  documents=df_docs[['id','title','body']].copy()
  documents['similarity']=df_docs['vector'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
  ranked_docs = documents.sort_values(by='similarity',ascending=False,inplace=False)
  
  return ranked_docs.reset_index(drop=True)

retireved_df = ranking_ir(df_quers['query'][0])
print('retrieved_df')
print(retireved_df)
# print(f"len(df_quers['query']) = {len(df_quers['query'])}")
# print(retireved_df['id'].to_list())

qrels = json.load(open("cranfield/cran_qrels.json", 'r'))[:]
query_ids, queries = [item["query number"] for item in quers_json], \
								[item["query"] for item in quers_json]
doc_IDs_ordered = []

for i in range(len(df_quers['query'])):
	retrieved_df = ranking_ir(df_quers['query'][i])
	print(f'Ranked query {i}')
	doc_IDs_ordered.append(retrieved_df['id'].to_list())

precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		
for k in range(1, 11):
	precision = Evaluation().meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
	precisions.append(precision)

	recall = Evaluation().meanRecall(doc_IDs_ordered, query_ids, qrels, k)
	recalls.append(recall)
	
	fscore = Evaluation().meanFscore(doc_IDs_ordered, query_ids, qrels, k)
	fscores.append(fscore)
	print(f"k = {k}, Precision: {precision}, Recall: {recall} and F-score: {fscore}")
	
	MAP = Evaluation().meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
	MAPs.append(MAP)

	nDCG = Evaluation().meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
	nDCGs.append(nDCG)

	print(f"k = {k}, MAP: {MAP}, nDCG: {nDCG}")

plt.plot(range(1, 11), precisions, label="Precision")
plt.plot(range(1, 11), recalls, label="Recall")
plt.plot(range(1, 11), fscores, label="F-Score")
plt.plot(range(1, 11), MAPs, label="MAP")
plt.plot(range(1, 11), nDCGs, label="nDCG")
plt.legend()
plt.title("Evaluation Metrics - Cranfield Dataset")
plt.xlabel("k")
#plt.savefig("output/eval_plot_w2v.png")
plt.show()


print('Finished at: ', str(datetime.now()))

# body = df_docs["body"].values
# body = [nltk.word_tokenize(sentence) for sentence in body]
# #print(f'len(body) = {len(body[1])}')
# model = Word2Vec(sentences = body, size=50, min_count=1,window=5, sg=1, seed=42)
# #print(model.most_similar('heat'))
# print(body[0][0])
# print(model.similarity('heat', body[0][0]))

'''
flat_docs = list(itertools.chain.from_iterable(docs_txt))
#print(f'docs_txt = {docs_txt[:2]}')
flat_docs = list(set(list(itertools.chain.from_iterable(flat_docs))))
#print(f'flat_docs2 = {flat_docs[:5]}')
flat_quers = list(itertools.chain.from_iterable(queries_txt))
flat_quers = list(set(list(itertools.chain.from_iterable(flat_quers))))

flat = flat_docs + flat_quers
combined_data = list(set(flat))

print(f"Docs_list total = {len(flat_docs)}, unique = {len(list(set(flat_docs)))}")
print(f"Quers_list total = {len(flat_quers)}, unique = {len(list(set(flat_quers)))}")
print(f"Combined_data total = {len(flat)}, unique = {len(combined_data)}")

#print(f'combined_data = {combined_data[:100]}')

train_data=[]
for i in combined_data:
    train_data.append(i.split())

print(train_data[:5])
'''
