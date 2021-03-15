from util import *
 
import spacy
nlp = spacy.load('en_core_web_sm')


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		for i in range(len(text)):
			sentence = " ".join(text[i]) 

			doc = nlp(sentence)
			lemma_word = []

			for word in doc:
				lemma_word.append(word.lemma_)
			reducedText.append(lemma_word)

		return reducedText


