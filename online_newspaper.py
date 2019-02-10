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
from newspaper import news_pool

class Newspaper(object):

	def __init__(self, url):
		self.url = url

	def get_sources(self):
		return newspaper.build(self.url, memoize_articles=False)

	def print_cnn_articles(self):
		self.url = 'http://money.cnn.com'

		# first_newspaper = news.Newspaper(url=url)

		cnn = self.get_sources()

		sources = [cnn]

		news_pool.set(sources, threads_per_source=2)
		news_pool.join()

		for cnn_art in cnn.articles:
			if cnn_art.title:
				print(cnn_art.title)
				print(cnn_art.summary)