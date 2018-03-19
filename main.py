'''

Jacob McKenna
UAF Graduate Project
main.py - Starting point of the entire project.

'''
# File panda.py import
import panda
import news
import article


def main():
    url = 'http://money.cnn.com'

    first_newspaper = news.Newspaper(url=url)

    muh_articles = first_newspaper.get_articles()

    for article in muh_articles:
    	print(article.title)


if __name__ == "__main__":
    main()



