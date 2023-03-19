import sys
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

sys.path.append('Dataloader')
from dataloader import Dataloader

sys.path.append('config')
from settings import TRAIN_PERCENT
from settings import VAL_PERCENT
from settings import TEST_PERCENT
from settings import BATCH_SIZE

from feed_forward import FeedForwardModel

data = Dataloader('Data/data.csv')
train, val, test = data.split(TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT)

train_dataset = tf.data.Dataset.from_tensor_slices(
        (train[:, :2].reshape(-1, 1, 2), train[:, 2])
    ).batch(
            BATCH_SIZE
            ).prefetch(buffer_size=tf.data.AUTOTUNE)


val_dataset = tf.data.Dataset.from_tensor_slices(
        (val[:, :2].reshape(-1, 1, 2), val[:, 2])
    ).batch(
        BATCH_SIZE
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

model_name = 'FeedForward_v1'
model = FeedForwardModel(hidden_neurons=10)
model.compile(loss = 'mean_squared_error', metrics = ['mean_absolute_error'], optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))

checkpoint_dir = "./Artifacts/Models/" + model_name + "/Checkpoints/"
checkpoint_path = checkpoint_dir + "cp-{epoch:04d}.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             monitor='val_loss', 
                             verbose=1,
                             save_weights_only = True, 
                             mode='auto')

tf_path = "./Artifacts/Models/" + model_name + "/Model/tf"
fullModelSave = ModelCheckpoint(filepath=tf_path, 
                             monitor='val_loss', 
                             verbose=1,
                             save_best_only=True,
                             mode='auto')


log_dir = "./Artifacts/Models/" + model_name + "/Logs/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks_list = [checkpoint, tensorboard_callback, fullModelSave]


epochs = 1000
model.fit(
    train_dataset,
    epochs = epochs, 
    validation_data = val_dataset,
    callbacks = callbacks_list,
    verbose = 1)