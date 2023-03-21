import tensorflow as tf

def CascadeForwardModel(input_size = (1, 2), hidden_neurons = [10]):
    inputs = tf.keras.layers.Input(input_size)

    layers = [inputs]

    for neurons in hidden_neurons:
        concat = tf.keras.layers.Concatenate(axis=-1)(layers)
        hidden = tf.keras.layers.Dense(neurons, activation = 'relu')(concat)
        layers.append(hidden)

    concat = tf.keras.layers.Concatenate(axis=-1)(layers)
    outputs = tf.keras.layers.Dense(1, activation = 'relu')(concat)

    model = tf.keras.Model(inputs, outputs)

    return model