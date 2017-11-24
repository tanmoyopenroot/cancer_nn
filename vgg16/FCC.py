from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense
from keras import optimizers

import loadData

#contrib check
class FCC:


	def __init__(
		self,
		hidden_layers = 1,
		neurons = [64, 1], 
	):

		self.hidden_layers = hidden_layers
		self.neurons = neurons

	def loadFCC( self, input_shape ):

		model = Sequential()

		model.add( Flatten( input_shape = input_shape ) )

		if self.hidden_layers == len(self.neurons) - 1:

			for neuron in self.neurons[ : len(self.neurons) - 1 ]:
				model.add(Dense(
					neuron,
					activation = 'relu'
				))
				model.add( Dropout(0.5) )

			model.add( Dense( 
				self.neurons[ len(self.neurons) - 1 ],
				activation = "sigmoid" 
			))

			return model

		else: 
			print "layer and length of neurons list should be equal"

			return False

	
	def trainFCC( 
		self, 
		epochs, 
		lr,
		batch_size = 32, 
		loss = 'binary_crossentropy',
		train_data_path = "train_transfer_block3_pool_values.npy",
		vald_data_path = "validation_transfer_block3_pool_values.npy",
		mmap = 'r' # set to None if the dataset fits into memory
	):

		train_data, vald_data = loadData.data( train_data_path, vald_data_path, mmap)
		train_labels, vald_labels = loadData.labels()

		model = self.loadFCC( train_data.shape[1: ] )

		if model: 

			model.compile(
				loss = self.loss,
				optimizer = optimizers.Adam( lr ),
				metrics = ['accuracy']
			)

			history = model.fit(
				train_data, 
				train_labels,
				epochs = epochs,
				batch_size = batch_size,
				validation_data = ( vald_data, vald_labels ),
				shuffle = True
			)

			loadData.plotTraining( history )
		else:
			print "error in instantiating model"

	def testFCC( self ):

		vald_data = loadData.valdData( self.vald_data_path, mmap )
		_, vald_labels = loadData.labels()

		model = self.loadFCC( vald_data.shape )

		if model:
		   	pred = model.predict(

		        validation_data,
		        batch_size = 16,
		        verbose = 1
		    )

		else:
			print "error in instantiating model"







	