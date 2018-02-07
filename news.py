'''

Jacob McKenna
UAF Graduate Project
panda.py - Simple import and testing file for learning pandas.

***** 
File name needs to change when appropriate behavior is chosen!
*****

'''
import newspaper

cnn_paper = newspaper.build('https://www.cnn.com', memoize_articles=False)

for article in cnn_paper.articles:
	print(article.url)
