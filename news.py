'''

Jacob McKenna
UAF Graduate Project
news.py - Simple newspaper objects that contain only a URL. Builds
	the source object whne get_sources is called.

***** 
File name needs to change when appropriate behavior is chosen!
*****

'''
import newspaper


class Newspaper(object):

	def __init__(self, url):
		self.url = url

	def get_sources(self):
		return newspaper.build(self.url, memoize_articles=False)

	# def get_articles(self):
