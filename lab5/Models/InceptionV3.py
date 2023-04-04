import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, AveragePooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, concatenate

def conv_batch_normalization(prev_layer, filters, kernel_size, strides = (1, 1), padding = 'same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(prev_layer)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    return x



def StemBlock(prev_layer):
    x = conv_batch_normalization(prev_layer, filters = 32, kernel_size=(3,3), strides=(2,2))
    x = conv_batch_normalization(x, filters = 32, kernel_size=(3,3))
    x = conv_batch_normalization(x, filters = 64, kernel_size=(3,3))
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = conv_batch_normalization(x, filters = 80, kernel_size=(1,1))
    x = conv_batch_normalization(x, filters = 192, kernel_size=(3,3))
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    return x



def InceptionBlock_A(prev_layer, branch3_filters):
    branch1 = conv_batch_normalization(prev_layer, filters = 64, kernel_size=(1,1))
    branch1 = conv_batch_normalization(branch1, filters = 96, kernel_size=(3,3))
    branch1 = conv_batch_normalization(branch1, filters = 96, kernel_size=(3,3))
    
    branch2 = conv_batch_normalization(prev_layer, filters = 48, kernel_size=(1,1))
    branch2 = conv_batch_normalization(branch2, filters = 64, kernel_size=(3,3))
    
    branch3 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(prev_layer)
    branch3 = conv_batch_normalization(branch3, filters = branch3_filters, kernel_size=(1,1))
    
    branch4 = conv_batch_normalization(prev_layer, filters = 64, kernel_size=(1,1))
    
    output = concatenate([branch1 , branch2 , branch3 , branch4], axis=3)
    
    return output



def ReductionBlock_A(prev_layer):
    branch1 = conv_batch_normalization(prev_layer, filters = 64, kernel_size=(1,1))
    branch1 = conv_batch_normalization(branch1, filters = 96, kernel_size=(3,3))
    branch1 = conv_batch_normalization(branch1, filters = 96, kernel_size=(3,3), strides=(2,2))
    
    branch2 = conv_batch_normalization(prev_layer, filters = 384, kernel_size=(3,3), strides=(2,2))
    
    branch3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(prev_layer)
    
    output = concatenate([branch1 , branch2 , branch3], axis=3)
    
    return output



def InceptionBlock_B(prev_layer, branch1_branch2_filters):
    branch1 = conv_batch_normalization(prev_layer, filters = branch1_branch2_filters, kernel_size=(1,1))
    branch1 = conv_batch_normalization(branch1, filters = branch1_branch2_filters, kernel_size=(7,1))
    branch1 = conv_batch_normalization(branch1, filters = branch1_branch2_filters, kernel_size=(1,7))
    branch1 = conv_batch_normalization(branch1, filters = branch1_branch2_filters, kernel_size=(7,1))
    branch1 = conv_batch_normalization(branch1, filters = 192, kernel_size=(1,7))
    
    branch2 = conv_batch_normalization(prev_layer, filters = branch1_branch2_filters, kernel_size=(1,1))
    branch2 = conv_batch_normalization(branch2, filters = branch1_branch2_filters, kernel_size=(1,7))
    branch2 = conv_batch_normalization(branch2, filters = 192, kernel_size=(7,1))
    
    branch3 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(prev_layer)
    branch3 = conv_batch_normalization(branch3, filters = 192, kernel_size=(1,1))
    
    branch4 = conv_batch_normalization(prev_layer, filters = 192, kernel_size=(1,1))
    
    output = concatenate([branch1 , branch2 , branch3 , branch4], axis=3)
    
    return output



def ReductionBlock_B(prev_layer):
    branch1 = conv_batch_normalization(prev_layer, filters = 192, kernel_size=(1,1))
    branch1 = conv_batch_normalization(branch1, filters = 192, kernel_size=(1,7))
    branch1 = conv_batch_normalization(branch1, filters = 192, kernel_size=(7,1))
    branch1 = conv_batch_normalization(branch1, filters = 192, kernel_size=(3,3), strides=(2,2), padding='valid')
    
    branch2 = conv_batch_normalization(prev_layer, filters = 192, kernel_size=(1,1))
    branch2 = conv_batch_normalization(branch2, filters = 320, kernel_size=(3,3), strides=(2,2), padding='valid')
    
    branch3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(prev_layer)
    
    output = concatenate([branch1 , branch2 , branch3], axis=3)
    
    return output



def InceptionBlock_C(prev_layer):
    branch1 = conv_batch_normalization(prev_layer, filters = 448, kernel_size=(1,1))
    branch1 = conv_batch_normalization(branch1, filters = 384, kernel_size=(3,3))
    branch1_1 = conv_batch_normalization(branch1, filters = 384, kernel_size=(1,3))
    branch1_2 = conv_batch_normalization(branch1, filters = 384, kernel_size=(3,1))
    branch1 = concatenate([branch1_1, branch1_2], axis=3)
    
    branch2 = conv_batch_normalization(prev_layer, filters = 384, kernel_size=(1,1))
    branch2_1 = conv_batch_normalization(branch2, filters = 384, kernel_size=(1,3))
    branch2_2 = conv_batch_normalization(branch2, filters = 384, kernel_size=(3,1))
    branch2 = concatenate([branch2_1, branch2_2], axis=3)
    
    branch3 = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(prev_layer)
    branch3 = conv_batch_normalization(branch3, filters = 192, kernel_size=(1,1))
    
    branch4 = conv_batch_normalization(prev_layer, filters = 320, kernel_size=(1,1))
    
    output = concatenate([branch1 , branch2 , branch3 , branch4], axis=3)
    
    return output



def AuxClassifier(prev_layer, output_size):
    x = AveragePooling2D(pool_size=(5,5), strides=(3,3))(prev_layer)
    x = conv_batch_normalization(x, filters = 128, kernel_size=(1,1))
    x = Flatten()(x)
    x = Dense(768, activation='relu')(x)
    x = Dropout(0.2)(x)
    if(output_size == 1):
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(output_size, activation='softmax')(x)
    return x



def InceptionV3(input_size, output_size):
    input_layer = Input(shape=input_size)
    
    x = StemBlock(input_layer)
    
    x = InceptionBlock_A(x, branch3_filters = 32)
    x = InceptionBlock_A(x, branch3_filters = 64)
    x = InceptionBlock_A(x, branch3_filters = 64)
    
    x = ReductionBlock_A(x)
    
    x = InceptionBlock_B(x, branch1_branch2_filters=128)
    x = InceptionBlock_B(x, branch1_branch2_filters=160)
    x = InceptionBlock_B(x, branch1_branch2_filters=160)
    x = InceptionBlock_B(x, branch1_branch2_filters=192)
    
    Aux = AuxClassifier(x, output_size)
    
    x = ReductionBlock_B(x)
    
    x = InceptionBlock_C(x)
    x = InceptionBlock_C(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.2)(x)
    if(output_size == 1):
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(output_size, activation='softmax')(x)
    
    model = Model(input_layer, [x, Aux])
    
    return model