import tensorflow as tf

def Model(hidden_neurons):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(28,28)))
    model.add(tf.keras.layers.Flatten())
    for neurons in hidden_neurons:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model