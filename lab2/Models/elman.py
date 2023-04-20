import tensorflow as tf

def ElmanModel(input_size = (2), hidden_neurons = [10]):
    inputs = tf.keras.layers.Input(input_size)

    current_layer = tf.expand_dims(inputs, axis = 1)
    current_layer = tf.keras.layers.SimpleRNN(hidden_neurons[0])(current_layer)

    for neurons in hidden_neurons[1:]:
        current_layer = tf.expand_dims(current_layer, axis = 1)
        current_layer = tf.keras.layers.SimpleRNN(neurons, activation='relu')(current_layer)

    outputs = tf.keras.layers.Dense(1, activation = 'relu')(current_layer)

    model = tf.keras.Model(inputs, outputs)

    return model