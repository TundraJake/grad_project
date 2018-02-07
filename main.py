'''

Jacob McKenna
UAF Graduate Project
main.py - Starting point of the entire project.

'''
# File panda.py import
import panda
import news
import article
from newspaper import Article


def main():
    url = 'http://money.cnn.com/2018/02/05/investing/stock-market-today-dow-jones/index.html'
    first_article = article.Article(url=url)

    first_article.print_title()
    first_article.print_keywords()
    first_article.print_text()

if __name__ == "__main__":
    main()



