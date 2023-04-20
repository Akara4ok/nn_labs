import tensorflow as tf

def FeedForwardModel(input_size = (2), hidden_neurons = [10]):
    inputs = tf.keras.layers.Input(input_size)
    current_layer = inputs
    for neurons in hidden_neurons:
        current_layer = tf.keras.layers.Dense(neurons, activation = 'relu')(current_layer)
    outputs = tf.keras.layers.Dense(1, activation = 'relu')(current_layer)

    model = tf.keras.Model(inputs, outputs)

    return model