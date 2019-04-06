# For building model,I mainly used keras
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dropout
class ar_model:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		cnn_model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# update the input shape and channels dimension when using channels first
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1




                #--------architecture--------#
		cnn_model.add(Conv2D(16, (3, 3), padding="same",input_shape=inputShape))
		cnn_model.add(Activation("relu"))#Using rectified linear unit
		cnn_model.add(BatchNormalization(axis=chanDim))
		cnn_model.add(Conv2D(16, (3, 3), padding="same"))
		cnn_model.add(Activation("relu"))#Using rectified linear unit
		cnn_model.add(BatchNormalization(axis=chanDim))
		cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
		cnn_model.add(Dropout(0.25))     #Using dropout
		cnn_model.add(Conv2D(32, (3, 3), padding="same"))
		cnn_model.add(Activation("relu"))#Using rectified linear unit
		cnn_model.add(BatchNormalization(axis=chanDim))
		cnn_model.add(Conv2D(32, (3, 3), padding="same"))
		cnn_model.add(Activation("relu"))#Using rectified linear unit
		cnn_model.add(BatchNormalization(axis=chanDim))
		cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
		cnn_model.add(Dropout(0.25))    #Using dropout
		cnn_model.add(Flatten())        #Mapping in same dimention
		cnn_model.add(Dense(64))        #Fully connected layer
		cnn_model.add(Activation("relu"))
		cnn_model.add(BatchNormalization())
		cnn_model.add(Dropout(0.5))     #Using dropout
		cnn_model.add(Dense(classes))
		cnn_model.add(Activation("softmax")) #Mapping data to possibilities
		return cnn_model
