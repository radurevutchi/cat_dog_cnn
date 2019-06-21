from keras.models import Sequential # initializes NN as sequanential
from keras.layers import Conv2D # 2D Conv. layer (3D used for video and time dimension)
from keras.layers import MaxPooling2D # max pooling step
from keras.layers import Flatten # flattens 2D images/2d arrays to a vector
from keras.layers import Dense
from keras.models import model_from_json, load_model
from keras.preprocessing.image import ImageDataGenerator


# making predictions
import numpy as np
from keras.preprocessing import image


'''
# creates sequential NN object
classifier = Sequential()


# adding a convolutional layer
# takes in 4 arguments
# first argument is number of filters to use
# second is the size of each filter
# third is the size of the input 64x64 pixels 3=RGB
# fourth is the activation function (relu is standard in CNN)
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))


# adding a max pooling layer (will take max value from each 4x4)
# if stride not assigned, it defaults to pool_size
classifier.add(MaxPooling2D(pool_size=(2,2)))


# flattening step
classifier.add(Flatten())


# adds a dense layer after the flattened layer
classifier.add(Dense(units=128, activation='relu'))


# this is the output layer/node cause there are only 2 classes
classifier.add(Dense(units=1, activation='sigmoid'))

# compiles the cnn and tells it how to train
# 3 arguments
# optimizer is gradient descent
# loss parameter is the loss function
# metrics is the performance metric (how to determine improvement/accuracy)
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])




# PREPROCESSING IMAGES

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('test_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')



print("TRAINING MODEL")
# steps_per_epoch is number of training imgages
classifier.fit_generator(training_set,
                            steps_per_epoch=1589,
                            epochs=4,
                            validation_data=test_set,
                            validation_steps=380)




# saving the model and weights
classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)

classifier.save_weights("model1.h5")

'''

# loading the model and weights

with open("model1.json",'r') as json_file:
    classifier = model_from_json(json_file.read())

classifier.load_weights("model1.h5")






print("TESTING MODEL")
# testing an image

for i in range(1,5):
    test_image = image.load_img("my_set/cats/cat" +str(i) + ".jpg",
                                target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print("This classifier predicted: " + prediction)

for i in range(1,5):
    test_image = image.load_img("my_set/dogs/dog" +str(i) + ".jpg",
                                target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print("This classifier predicted: " + prediction)
