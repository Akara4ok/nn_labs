import tensorflow as tf

def FeedForwardModel(input_size = (1, 2), hidden_neurons = 10):
    inputs = tf.keras.layers.Input(input_size)
    hidden = tf.keras.layers.Dense(hidden_neurons, activation = 'relu')(inputs)
    outputs = tf.keras.layers.Dense(1, activation = 'relu')(hidden)

    model = tf.keras.Model(inputs, outputs)

    return model