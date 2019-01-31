'''

Jacob McKenna
UAF Graduate Project
nn.py - Keras NN market prediction.

'''
# Create first network with Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence, one_hot
from keras.models import Sequential
from keras.layers import Dense
import numpy.random as npr
import unittest
import nltk
# import psycopg2

# nltk.download('punkt')
# from nltk.tokenize import word_tokenize

# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()



# CONN = psycopg2.connect(host='localhost', user='stock', password='money', dbname='stock_market_data')
# CURS = CONN.cursor()

# SELECT_QUERY = "SELECT text FROM \"Tweets\" limit 10;" 

npr.seed(1)
# fix random seed for reproducibility

class Test(unittest.TestCase):

	def print_break(self):
		print()
		print()
		print()
		print()

	def get_numpy_shape(self, np_arr):
		return np_arr.shape

	def test_simple_keras_one_hot(self):
		# define the document
		text = 'The quick brown fox jumped over the lazy dog.'
		
		# estimate the size of the vocabulary
		words = set(text_to_word_sequence(text))
		vocab_size = len(words)
		print(vocab_size)
		
		# integer encode the document
		result = one_hot(text, round(vocab_size*1.3))
		print(result)
		self.print_break()


	def test_nn_(self):


		docs = ["Where do random thoughts come from?",
			"I hear that Nancy is very pretty.",
			"Sometimes it is better to just walk away from things and go back to them later when you’re in a better frame of mind.",
			"The clock within this blog and the clock on my laptop are 1 hour different from each other.",
			"I want more detailed information.",
			"She was too short to see over the fence.",
			"The river stole the gods.",
			"Writing a list of random sentences is harder than I initially thought it would be.",
			"How was the math test?",
			"Tom got a small piece of pie.",
			"The shooter says goodbye to his love.",
			"The body may perhaps compensates for the loss of a true metaphysics.",
			"I am happy to take your donation; any amount will be greatly appreciated.",
			"Hurry!",
			"Rock music approaches at high velocity.",
			"She only paints with bold colors; she does not like pastels.",
			"Mary plays the piano.",
			"Is it free?",
			"Italy is my favorite country; in fact, I plan to spend two weeks there next year.",
			"I was very proud of my nickname throughout high school but today- I couldn’t be any different to what my nickname was.",
			"If you like tuna and tomato sauce- try combining the two. It’s really not as bad as it sounds.",
			"I will never be this young again. Ever. Oh damn… I just got older.",
			"She folded her handkerchief neatly.",
			"We have a lot of rain in June.",
			"The mysterious diary records the voice.",
			"Joe made the sugar cookies; Susan decorated them.",
			"I want to buy a onesie… but know it won’t suit me.",
			"Sometimes, all you need to do is completely make an ass of yourself and laugh it off to realise that life isn’t so bad after all.",
			"The lake is a long way from here.",
			"I love eating toasted cheese and tuna sandwiches.",
			"This is a Japanese doll.",
			"The book is in front of the table.",
			"A purple pig and a green donkey flew a kite in the middle of the night and ended up sunburnt.",
			"Wow, does that work?",
			"I currently have 4 windows open up… and I don’t know why.",
			"I am never at home on Sundays.",
			"Wednesday is hump day, but has anyone asked the camel if he’s happy about it?",
			"He didn’t want to go to the dentist, yet he went anyway.",
			"Abstraction is often one floor above you.",
			"He told us a very exciting adventure story.",
			"Everyone was busy, so I went to the movie alone.",
			"The waves were crashing on the shore; it was a lovely sight.",
			"Should we start class now, or should we wait for everyone to get here?",
			"I would have gotten the promotion, but my attendance wasn’t good enough.",
			"Lets all be unique together until we realise we are all the same.",
			"Cats are good pets, for they are clean and are not noisy.",
			"This is the last random sentence I will be writing and I am going to stop mid-sent",
			"I'd rather be a bird than a fish.",
			"Please wait outside of the house.",
			"The sky is clear; the stars are twinkling.",
			"I checked to make sure that he was still alive.",
			"He turned in the research paper on Friday; otherwise, he would have not passed the class.",
			"What was the person thinking when they discovered cow’s milk was fine for human consumption… and why did they do it in the first place!?",
			"Check back tomorrow; I will see if the book has arrived.",
			"Don't step on the broken glass.",
			"The quick brown fox jumps over the lazy dog.",
			"If I don’t like something, I’ll stay away from it.",
			"She always speaks to him in a loud voice.",
			"She works two jobs to make ends meet; at least, that was her reason for not having time to join us.",
			"There were white out conditions in the town; subsequently, the roads were impassable.",
			"Someone I know recently combined Maple Syrup & buttered Popcorn thinking it would taste like caramel popcorn. It didn’t and they don’t recommend anyone else do it either.",
			"The memory we used to share is no longer coherent.",
			"We need to rent a room for our party.",
			"I often see the time 11:11 or 12:34 on clocks.",
			"The old apple revels in its authority.",
			"She borrowed the book from him many years ago and hasn't yet returned it.",
			"Malls are great places to shop; I can find everything I need under one roof.",
			"She did her best to help him.",
			"A glittering gem is not enough.",
			"Sixty-Four comes asking for bread.",
			"Last Friday in three week’s time I saw a spotted striped blue worm shake hands with a legless lizard.",
			"My Mum tries to be cool by saying that she likes all the same things that I do.",
			"They got there early, and they got really good seats.",
			"He said he was not there yesterday; however, many people saw him there.",
			"She wrote him a long letter, but he didn't read it.",
			"If the Easter Bunny and the Tooth Fairy had babies would they take your teeth and leave chocolate for you?",
			"Yeah, I think it's a good environment for learning English.",
			"If Purple People Eaters are real… where do they find purple people to eat?",
			"I really want to go to work, but I am too sick to drive.",
			"She advised him to come back at once.",
			"A song can make or ruin a person’s day if they let it get to them.",
			"We have never been to Asia, nor have we visited Africa.",
			"I am counting my calories, yet I really want dessert.",
			"He ran out of money, so he had to stop playing poker.",
			"I think I will buy the red car, or I will lease the blue one.",
			"Two seats were vacant.",
			"She did not cheat on the test, for it was not the right thing to do.",
			"It was getting dark, and we weren’t there yet.",
			"The stranger officiates the meal.",
			"There was no ice cream in the freezer, nor did they have money to go to the store.",
			"Let me help you with your baggage.",
			"When I was little I had a car door slammed shut on my hand. I still remember it quite vividly.",
			"Christmas is coming.",
			"The waves were crashing on the shore; it was a lovely sight.",
			"If Purple People Eaters are real… where do they find purple people to eat?",
			"Malls are great places to shop; I can find everything I need under one roof.",
			"She advised him to come back at once.",
			"He ran out of money, so he had to stop playing poker.",
			"Hurry!",
			"My Mum tries to be cool by saying that she likes all the same things that I do.",
			"Sometimes it is better to just walk away from things and go back to them later when you’re in a better frame of mind.",
			"He didn’t want to go to the dentist, yet he went anyway.",
			"I love eating toasted cheese and tuna sandwiches.",
			"She wrote him a long letter, but he didn't read it.",
			"She folded her handkerchief neatly.",
			"A glittering gem is not enough.",
			"When I was little I had a car door slammed shut on my hand. I still remember it quite vividly.",
			"I often see the time 11:11 or 12:34 on clocks.",
			"I was very proud of my nickname throughout high school but today- I couldn’t be any different to what my nickname was.",
			"I hear that Nancy is very pretty.",
			"Don't step on the broken glass.",
			"Yeah, I think it's a good environment for learning English.",
			"I want to buy a onesie… but know it won’t suit me.",
			"Joe made the sugar cookies; Susan decorated them.",
			"Sixty-Four comes asking for bread.",
			"She borrowed the book from him many years ago and hasn't yet returned it.",
			"Christmas is coming.",
			"I really want to go to work, but I am too sick to drive.",
			"He told us a very exciting adventure story.",
			"Should we start class now, or should we wait for everyone to get here?",
			"It was getting dark, and we weren’t there yet.",
			"Writing a list of random sentences is harder than I initially thought it would be.",
			"Italy is my favorite country; in fact, I plan to spend two weeks there next year.",
			"Where do random thoughts come from?",
			"She was too short to see over the fence.",
			"The river stole the gods.",
			"She did not cheat on the test, for it was not the right thing to do.",
			"They got there early, and they got really good seats.",
			"The stranger officiates the meal.",
			"I'd rather be a bird than a fish.",
			"The body may perhaps compensates for the loss of a true metaphysics.",
			"Is it free?",
			"I am counting my calories, yet I really want dessert.",
			"I will never be this young again. Ever. Oh damn… I just got older.",
			"There were white out conditions in the town; subsequently, the roads were impassable.",
			"The old apple revels in its authority.",
			"The quick brown fox jumps over the lazy dog.",
			"Let me help you with your baggage.",
			"Last Friday in three week’s time I saw a spotted striped blue worm shake hands with a legless lizard.",
			"A song can make or ruin a person’s day if they let it get to them.",
			"There was no ice cream in the freezer, nor did they have money to go to the store.",
			"I would have gotten the promotion, but my attendance wasn’t good enough.",
			"Wednesday is hump day, but has anyone asked the camel if he’s happy about it?",
			"Two seats were vacant.",
			"We have never been to Asia, nor have we visited Africa.",
			"If you like tuna and tomato sauce- try combining the two. It’s really not as bad as it sounds.",
			"Abstraction is often one floor above you.",
			"A purple pig and a green donkey flew a kite in the middle of the night and ended up sunburnt.",
			"Rock music approaches at high velocity.",
			"She works two jobs to make ends meet; at least, that was her reason for not having time to join us.",
			"We have a lot of rain in June.",
			"I currently have 4 windows open up… and I don’t know why.",
			"She only paints with bold colors; she does not like pastels.",
			"The lake is a long way from here.",
			"I am happy to take your donation; any amount will be greatly appreciated.",
			"Everyone was busy, so I went to the movie alone.",
			"He turned in the research paper on Friday; otherwise, he would have not passed the class.",
			"I checked to make sure that he was still alive.",
			"The clock within this blog and the clock on my laptop are 1 hour different from each other.",
			"The book is in front of the table.",
			"Wow, does that work?",
			"The mysterious diary records the voice.",
			"I am never at home on Sundays.",
			"If the Easter Bunny and the Tooth Fairy had babies would they take your teeth and leave chocolate for you?",
			"Someone I know recently combined Maple Syrup & buttered Popcorn thinking it would taste like caramel popcorn. It didn’t and they don’t recommend anyone else do it either.",
			"She did her best to help him.",
			"The shooter says goodbye to his love.",
			"The sky is clear; the stars are twinkling.",
			"I think I will buy the red car, or I will lease the blue one.",
			"Cats are good pets, for they are clean and are not noisy.",
			"We need to rent a room for our party.",
			"The memory we used to share is no longer coherent.",
			"Please wait outside of the house.",
			"How was the math test?",
			"This is a Japanese doll.",
			"Check back tomorrow; I will see if the book has arrived.",
			"Lets all be unique together until we realise we are all the same.",
			"What was the person thinking when they discovered cow’s milk was fine for human consumption… and why did they do it in the first place!?",
			"Mary plays the piano.",
			"Tom got a small piece of pie.",
			"He said he was not there yesterday; however, many people saw him there.",
			"This is the last random sentence I will be writing and I am going to stop mid-sent",
			"Sometimes, all you need to do is completely make an ass of yourself and laugh it off to realise that life isn’t so bad after all.",
			"She always speaks to him in a loud voice.",
			"I want more detailed information.",
			"If I don’t like something, I’ll stay away from it."
			]

		# create the tokenizer
		t = Tokenizer()
		
		# fit the tokenizer on the documents
		t.fit_on_texts(docs)
		
		# summarize what was learned
		print("Words counts")
		# print(t.word_counts)
		# print(t.document_count)
		print(t.word_index)
		# print(t.word_docs)
		
		# integer encode documents
		encoded_docs = t.texts_to_matrix(docs, mode='count')
	
		print(encoded_docs.shape)
		self.print_break()

		model = Sequential()
		model.add(Dense())









if __name__ == '__main__':
	unittest.main()