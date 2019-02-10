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
		super().__init__(url)
		self.complete_build()

	# Sets object for use.
	def complete_build(self):
		# self.build()
		self.download()
		self.parse()
		self.nlp()

	###### Debug ######
	def print_newlines(self):
		print('\n'*3)

	def print_keywords(self):
		print(self.keywords)
		self.print_newlines()

	def print_title(self):
		print(self.title)
		self.print_newlines()

	def print_summary(self):
		print(self.summary)
		self.print_newlines()
	
	def print_text(self):
		print(self.text)
		self.print_newlines()