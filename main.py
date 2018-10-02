'''

Jacob McKenna
UAF Graduate Project
main.py - Starting point of the entire project.

'''
# File panda.py import
import panda
import news
import article
from newspaper import news_pool


def main():
    url = 'http://money.cnn.com'

    first_newspaper = news.Newspaper(url=url)

    cnn = first_newspaper.get_sources()

    sources = [cnn]

    news_pool.set(sources, threads_per_source=2)
    news_pool.join()

    for cnn_art in cnn.articles:
        if cnn_art.title:
            print(cnn_art.title)
            print(cnn_art.summary)

if __name__ == "__main__":
    main()



