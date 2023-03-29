import sys
import numpy as np
import argparse
import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from keras.callbacks import ModelCheckpoint

sys.path.append('Dataloader')
from dataloader import Dataloader

sys.path.append('Models')
from feed_forward import FeedForwardModel
from cascade_forward import CascadeForwardModel
from elman import ElmanModel

sys.path.append('config')
from settings import TRAIN_PERCENT
from settings import VAL_PERCENT
from settings import TEST_PERCENT
from settings import DATA_PATH
from settings import BATCH_SIZE
from settings import SAVE_FOLDER
from settings import EPOCHS
from settings import RANDOM_SEED
from settings import DEFAULT_LR

def train(model_name,
          version,
          hidden_neurons,
          data_path = DATA_PATH,
          train_percent = TRAIN_PERCENT,
          val_percent = VAL_PERCENT,
          test_percent = TEST_PERCENT,
          save_folder = SAVE_FOLDER,
          epochs = EPOCHS,
          batch_size = BATCH_SIZE,
          lr = DEFAULT_LR):

    data = Dataloader(data_path)
    train, val, test = data.split(train_percent, val_percent, test_percent)
    
    model = None
    if(model_name == 'FeedForward'):
        model = FeedForwardModel(hidden_neurons=hidden_neurons)
    elif(model_name == 'CascadeForward'):
        model = CascadeForwardModel(hidden_neurons=hidden_neurons)
    elif(model_name == 'Elman'):
        model = ElmanModel(hidden_neurons=hidden_neurons)

    #values for schedules
    initial_learning_rate = 10**(-3)
    final_learning_rate = 10**(-7)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
    steps_per_epoch = int(len(train)/batch_size)

    learning_rate = lr
    if(lr == -1):
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=initial_learning_rate,
                        decay_steps=steps_per_epoch,
                        decay_rate=learning_rate_decay_factor
                    )


    model.compile(loss = 'mean_squared_error', metrics = ['mean_absolute_error'], 
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=learning_rate))
    
    path_to_save = save_folder + '/' + model_name + '/' + version + '/'

    checkpoint_dir = path_to_save + "Checkpoints/"
    checkpoint_path = checkpoint_dir + "cp-{epoch:04d}.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                monitor='val_loss', 
                                verbose=1,
                                save_weights_only = True, 
                                mode='auto')

    tf_path = path_to_save + "Model/tf"
    fullModelSave = ModelCheckpoint(filepath=tf_path, 
                                monitor='val_loss', 
                                verbose=1,
                                save_best_only=True,
                                mode='auto')


    log_dir = path_to_save + "Logs/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_list = [checkpoint, tensorboard_callback, fullModelSave]

    model.fit(
        np.reshape(train[:, :2], (-1, 2)),
        train[:, 2],
        batch_size,
        epochs = epochs, 
        validation_data = (np.reshape(val[:, :2], (-1, 2)), val[:, 2]),
        callbacks = callbacks_list,
        verbose = 1)
    
if __name__ == '__main__':
    tf.random.set_seed(RANDOM_SEED)

    parser=argparse.ArgumentParser()

    parser.add_argument("--model-name", "-m", help="type of model architecture", type=str)
    parser.add_argument("--version", "-v", help="version of the model", type=str)
    parser.add_argument("--hidden-neurons", "-n", help="array of hidden neurons(separated by comma)", type=str)

    parser.add_argument("--data-path", "-d", default=DATA_PATH, help="path to data", type=str)
    parser.add_argument("--save-folder", "-s", default=SAVE_FOLDER, help="path to save output", type=str)

    parser.add_argument("--train-percent", default=TRAIN_PERCENT, help="percent of training data", type=float)
    parser.add_argument("--val-percent", default=VAL_PERCENT, help="percent of validation data", type=float)
    parser.add_argument("--test-percent", default=TEST_PERCENT, help="percent of testing data", type=float)

    parser.add_argument("--epochs", "-e", default=EPOCHS, help="number of epochs", type=int)
    parser.add_argument("--batch-size", "-b", default=BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--learning-rate", "-l", default=DEFAULT_LR, help="learning rate(-1 -- means schedules)", type=float)

    args = vars(parser.parse_args())
    

    train(args['model_name'],
          args['version'],
          [int(x) for x in args['hidden_neurons'].split(',')],
          args['data_path'],
          args['train_percent'],
          args['val_percent'],
          args['test_percent'],
          args['save_folder'],
          args['epochs'],
          args['batch_size'],
          args['learning_rate'])