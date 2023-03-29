import tensorflow as tf

def ElmanModel(input_size = (1, 2), hidden_neurons = [10]):
    inputs = tf.keras.layers.Input(input_size)

    current_layer = inputs
    current_layer = tf.keras.layers.SimpleRNN(hidden_neurons[0])(current_layer)

    for neurons in hidden_neurons[1:]:
        current_layer = tf.keras.layers.Dense(neurons, activation='relu')(current_layer)

    outputs = tf.keras.layers.Dense(1, activation = 'relu')(current_layer)

    model = tf.keras.Model(inputs, outputs)

    return model