


from keras.models import Sequential # initializes NN as sequanential
from keras.layers import Conv2D # 2D Conv. layer (3D used for video and time dimension)
from keras.layers import MaxPooling2D # max pooling step
from keras.layers import Flatten # flattens 2D images/2d arrays to a vector
from keras.layers import Dense


# creates sequential NN object
classifier = Sequential()


# adding a convolutional layer
# 
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
