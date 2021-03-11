from util import *

# Add your import statements here
 
# import nltk
# from nltk.stem import WordNetLemmatizer
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
		# for i in range(len(text)):
		# 	sentence=text[i]
		# 	print("Type of sentence=text[i]: ", type(sentence))
		# 	word_list = []
		# 	for token in sentence:
		# 		print("Type of token.lemma_: ", type(token.lemma_))
		# 		word_list.append(token.lemma_)
		# 	reducedText.append(word_list)
		# return reducedText
		#Fill in code here
		for i in range(len(text)):
			# print("text[i][0]", text[i][0])
			# print("len(text[i]", len(text[i][0]))
			sentence = " ".join(text[i]) #[j] for j in len(text[i])])
			#print("sentence", sentence)
			doc = nlp(sentence)

			lemma_word = []
			# lemmatizer = WordNetLemmatizer()
			# for w in text[i]:
    		# 	word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
    		# 	word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
    		# 	word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
    		# 	lemma_word.append(word3)

			for word in doc:
				lemma_word.append(word.lemma_)
			reducedText.append(lemma_word)

		return reducedText


