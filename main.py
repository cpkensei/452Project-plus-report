from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import sys
import cv2
categories = ['Fake', 'Real']
#To test my code and model, directly run this script
#At line 23 0.png is real face, 1.png is fake face


def prepare(path):
    img_size = 50
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


#model = tf.keras.models.load_model('64x3-CNN.model')

#prediction = model.predict([prepare('dog.jpg')])

#print(prediction)
path_image = '1.png'
model = load_model('live_model.h5')

image = cv2.imread(path_image)#read image
image_d = cv2.resize(image, (32, 32))#64*64 pexils
image_d = image_d.astype("float") / 255.0
x = img_to_array(image_d)
x = np.expand_dims(x, axis=0)#use numpy array
preds = model.predict(x)[0]#load model to give percentage of each emotion
emotion_probability = np.max(preds)#give emotion back
label = categories[preds.argmax()]#give emotion name back
for (i, (emotion, prob)) in enumerate(zip(categories, preds)):
    # construct the label text
    text = "{}: {:.2f}%".format(emotion, prob * 100)
    print(text)
