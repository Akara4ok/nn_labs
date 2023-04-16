import tensorflow as tf
from keras import Model
from keras.layers import Input, Reshape, Conv2D, BatchNormalization, ReLU, GRU, Bidirectional, Dropout, Dense

def Speech2Text(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    input = Input((None, input_dim))
    x = Reshape((-1, input_dim, 1))(input)
    x = Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    for i in range(1, rnn_layers + 1):
        recurrent = GRU(
            units=rnn_units,
            recurrent_activation="sigmoid",
            return_sequences=True,
            reset_after=True
        )
        x = Bidirectional(
            recurrent, merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = Dropout(rate=0.5)(x)
    x = Dense(units=rnn_units * 2)(x)
    x = ReLU()(x)
    x = Dropout(rate=0.5)(x)
    output = Dense(units=output_dim + 1, activation="softmax")(x)
    model = Model(input, output)
    return model