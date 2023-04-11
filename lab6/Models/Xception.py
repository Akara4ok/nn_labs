import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense

def Xception(input_size, output_size):
    input_layer = Input(shape=input_size)
    output = Dense(output_size, activation='softmax')(input_layer)
    model = Model(input_layer, output)
    return model