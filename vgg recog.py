from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
from numpy.random import randint
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg16
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
top_model = model.output

top_model = Dense(512, activation = 'relu')(top_model)

top_model = Flatten()(top_model)

top_model = Dense(2, activation = 'softmax')(top_model)

model.input

new = Model(inputs = model.input, output = top_model)

new.summary()

new.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_set = train_datagen.flow_from_directory(
        'C:/Users/tjain/Documents/MLOps/data/train/',
        target_size=(224, 224),
        #class_mode='binary'
        batch_size=32,
        )
test_set = test_datagen.flow_from_directory(
        'C:/Users/tjain/Documents/MLOps/data/test/',
        target_size=(224, 224),
        #class_mode='binary'
        batch_size=32,
        )
new.fit(
        train_set,
        steps_per_epoch=140,
        epochs=1,
        validation_data=test_set,
        validation_steps=60)

new.save('myvgg.h5')


scores = new.evaluate(test_set, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

def load_image():
    i = randint(30)
    #img = cv2.imread("data/test/tushar/tushar" +str(i)+".jpg")
    img = Image.open("data/test/anima/anima"+str(i)+".jpg" )
    #img = Image.open("data/test/tushar/tushar"+str(i)+".jpg" )
    width = 224
    height = 224
    dim = (width, height)
    # resize image
    img = img.resize(dim)
    #img = resized.reshape(1, 28, 28, 1)
    img = np.expand_dims(img, axis =0)
    
    i += 1
    return img

y = 'tushar'



new.predict(load_image())




train_set.class_indices





