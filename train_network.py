#Import packages from sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#Preprocessing..
import numpy as np
import argparse
import cv2
import os
#Model file used
from mode.architecture import ar_model
#We mainly used keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths

# initialize the initial learning rate, batch size, and number of
# epochs to train for
init_learning_rate = 0.0001 #This is the learning rate
batch_size = 5 #This is the batch size
epoches = 5 #Numbers of epoches

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Your dataset")
ap.add_argument("-m", "--model", type=str, required=True,
	help="Your trained model")
args = vars(ap.parse_args())
print("loading fake and real faces' images from dataset.")
image_set = list(paths.list_images(args["dataset"]))
data = [] 
labels = []



for image_path in image_set:
	# get the image trom the dataset, load them and
	# resize training images to fixed 32x32 pixels
	label = image_path.split(os.path.sep)[-2]
	image = cv2.imread(image_path)
	image = cv2.resize(image, (32, 32))

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# transfer all images into a list
# data pre-processing
data = np.array(data, dtype="float") / 255.0
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(training_set_1, testing_set_1, training_set_2, testing_set_2) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# set up the optimizer and model
print("Step1: compiling CNN model...")
opt = Adam(lr=init_learning_rate, decay=init_learning_rate / epoches)
model = ar_model.build(width=32, height=32, depth=3,
	classes=len(label_encoder.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the convolutional neural network model
print("Step2: training network for ",epoches," epochs...")
Heuristic = model.fit_generator(aug.flow(training_set_1, training_set_2, batch_size=batch_size),
	validation_data=(testing_set_1, testing_set_2), steps_per_epoch=len(training_set_1) // batch_size,
	epochs=epoches)

# evaluate the convolutional neural network
print("Step3: evaluating network...")
predictions = model.predict(testing_set_1, batch_size=batch_size)
print(classification_report(testing_set_2.argmax(axis=1),
	predictions.argmax(axis=1), target_names=label_encoder.classes_))

# save the convolutional neural network model as h5/model/hdf5..
print("Step4: saving model...")
model.save(args["model"])


# when executing, type the following command to cmd
# cd to this file and type
# python train_network.py --dataset dataset --model live_model.h5

