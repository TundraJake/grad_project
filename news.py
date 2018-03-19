'''

Jacob McKenna
UAF Graduate Project
panda.py - Simple import and testing file for learning pandas.

***** 
File name needs to change when appropriate behavior is chosen!
*****

'''
import newspaper


class Newspaper(object):

	def __init__(self, url):
		self.url = url
		self.get_articles(self.url);

	def get_articles(self, url):
		return newspaper.build(self.url, memoize_articles=False)

