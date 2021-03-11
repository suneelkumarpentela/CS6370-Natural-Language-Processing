from util import *

# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer




class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""


		for sentence in text:
			tokens = text.strip().split(" ")
			tokens = [i for i in tokens if( (i.strip() != ".") and (i.strip() != ",") )]
			tokenizedText.append(tokens)

		#Fill in code here

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		
		tokenizedText = []

		tokenizer =  TreebankWordTokenizer()

		for sentence in text:
			tokens = tokenizer.tokenize(sentence.strip())
			tokens = [i for i in tokens if( (i.strip() != ".") and (i.strip() != ",") )]
			tokenizedText.append(tokens)
			

		#Fill in code here

		return tokenizedText