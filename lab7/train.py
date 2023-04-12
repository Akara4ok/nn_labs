import sys
import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

sys.path.append('Dataset')
import YelpDataset

sys.path.append('config')
import settings

sys.path.append('Models')
from lstmmodel import LstmModel

def train(version,
          data_path = settings.DATA_PATH,
          batch_size = settings.BATCH_SIZE,
          save_folder = settings.SAVE_FOLDER,
          epochs = settings.EPOCHS,
          lr = settings.DEFAULT_LR):
    
    train_ds, val_ds, test_ds = YelpDataset.load_data()
    
    model = LstmModel()
    
    model.compile(loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'], 
                  optimizer=tf.keras.optimizers.SGD(learning_rate=lr))
    
    path_to_save = save_folder  + '/' + version + '/'

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
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callbacks_list = [checkpoint, tensorboard_callback, fullModelSave]

    model.fit(
        train_ds,
        epochs = epochs, 
        shuffle=False,
        validation_data = val_ds,
        callbacks = callbacks_list,
        verbose = 1)


if __name__ == '__main__':
    tf.random.set_seed(settings.RANDOM_SEED)

    parser=argparse.ArgumentParser()
    
    parser.add_argument("--version", "-v", default="v1", help="version of the model", type=str)
    
    parser.add_argument("--data-path", "-d", default=settings.DATA_PATH, help="path to dataset", type=str)
    parser.add_argument("--save-folder", "-s", default=settings.SAVE_FOLDER, help="path to save models and logs", type=str)
    
    
    parser.add_argument("--epochs", "-e", default=settings.EPOCHS, help="number of epochs", type=int)
    parser.add_argument("--batch-size", "-b", default=settings.BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--learning-rate", "-l", default=settings.DEFAULT_LR, help="learning rate", type=float)

    args = vars(parser.parse_args())
    

    train(args['version'],
          args['data_path'],
          args['batch_size'],
          args['save_folder'],
          args['epochs'],
          args['learning_rate'])