import tensorflow as tf

def CascadeForwardModel(input_size = (2), hidden_neurons = [10]):
    inputs = tf.keras.layers.Input(input_size)

    concat = inputs

    for neurons in hidden_neurons:
        hidden = tf.keras.layers.Dense(neurons, activation = 'relu')(concat)
        concat = tf.keras.layers.Concatenate(axis=-1)([concat, hidden])

    outputs = tf.keras.layers.Dense(1, activation = 'relu')(concat)

    model = tf.keras.Model(inputs, outputs)

    return model