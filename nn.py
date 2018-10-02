'''

Jacob McKenna
UAF Graduate Project
nn.py - Keras NN market prediction.

'''
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import unittest

# fix random seed for reproducibility

class Test(unittest.TestCase):

	test_networks = []

	def test_adam_optimizer_nn(self):
		seed = 1
		numpy.random.seed(seed)
		# load pima indians dataset
		dataset = numpy.loadtxt("diab.csv", delimiter=",")
		# split into input (X) and output (Y) variables
		X = dataset[:,0:8]
		Y = dataset[:,8]
		# create model

		model = Sequential()
		model.add(Dense(8, input_dim=8, init='uniform', activation='relu'))

		model.add(Dense(32, init='uniform', activation='sigmoid'))
		model.add(Dense(64, init='uniform', activation='sigmoid'))
		model.add(Dense(22, init='uniform', activation='sigmoid'))
		# Output classifier, yes or no.
		model.add(Dense(1, init='uniform', activation='sigmoid'))


		# Compile model
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# Fit the model
		model.fit(X, Y, epochs=100, batch_size=10,  verbose=2)
		# calculate predictions
		predictions = model.predict(X)
		
		# round predictions
		rounded = [round(x[0]) for x in predictions]
		self.__class__.test_networks.append(rounded)

	def test_sgd_optimizer_nn(self):
		seed = 1
		numpy.random.seed(seed)
		# load pima indians dataset
		dataset = numpy.loadtxt("diab.csv", delimiter=",")
		# split into input (X) and output (Y) variables
		X = dataset[:,0:8]
		Y = dataset[:,8]
		# create model
		model = Sequential()
		model.add(Dense(8, input_dim=8, init='uniform', activation='relu'))
		model.add(Dense(32, init='uniform', activation='sigmoid'))
		# Output classifier, yes or no.
		model.add(Dense(1, init='uniform', activation='sigmoid'))
		# Compile model
		model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
		# Fit the model
		model.fit(X, Y, epochs=100, batch_size=10,  verbose=2)
		# calculate predictions
		predictions = model.predict(X)
		# round predictions
		rounded = [round(x[0]) for x in predictions]
		self.__class__.test_networks.append(rounded)




if __name__ == '__main__':
	unittest.main()