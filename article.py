'''

Jacob McKenna
UAF Graduate Project
panda.py - Simple import and testing file for learning pandas.

***** 
File name needs to change when appropriate behavior is chosen!
*****

'''
from newspaper import Article as Super_Article_Class

class Article(Super_Article_Class):

	def __init__(self, url):
		super(Article, self).__init__(url)
		self.complete_build()

	# 
	def complete_build(self):
		self.build()
		self.download()
		self.parse()
		self.nlp()

	def print_keywords(self):
		print(self.keywords)