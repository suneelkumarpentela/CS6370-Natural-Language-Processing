from util import *

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

class StopwordRemoval():

	def fromList(self, text,q=0):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		
		stopwordRemovedText = []
		stop_words = set(stopwords.words('english'))
		
		for i in range(len(text)):
			word_tokens = text[i]
			filtered_sentence = []
			for w in word_tokens:
				if w not in stop_words:
					filtered_sentence.append(w)
			stopwordRemovedText.append(filtered_sentence)

		if(q):
			final_text = stopwordRemovedText.copy()
			for i in stopwordRemovedText:
				t = []
				for j in i:
					t.append(j)
					k = 0
					for syn in wordnet.synsets(j):
						for l in syn.lemmas():
							if(k<6):
								if l.name() not in t:
									t.append(l.name())
									k+=1
				final_text.append(t)

		return stopwordRemovedText




	