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

sys.path.append('Preprocessing')
from preprocessing import YelpPreprocessing

sys.path.append('Callbacks')
from export_model import ExportModelCallback 

def train(version,
          data_path = settings.DATA_PATH,
          batch_size = settings.BATCH_SIZE,
          save_folder = settings.SAVE_FOLDER,
          epochs = settings.EPOCHS,
          lr = settings.DEFAULT_LR):
    
    train_ds, val_ds, test_ds = YelpDataset.load_data(data_path, val_percent=settings.VAL_PERCENT, buffer_size=settings.BUFFER_SIZE, batch_size=batch_size)
    preprocessor = YelpPreprocessing(train_ds)
    
    train_ds = train_ds.map(preprocessor.vectorize_text)
    val_ds = val_ds.map(preprocessor.vectorize_text)
    
    vocab_size = len(preprocessor.get_encoder().get_vocabulary())
    model = LstmModel(vocab_size)
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'], 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    
    path_to_save = save_folder  + '/' + version + '/'

    checkpoint_dir = path_to_save + "Checkpoints/"
    checkpoint_path = checkpoint_dir + "cp-{epoch:04d}.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                monitor='val_loss', 
                                verbose=1,
                                save_weights_only = True, 
                                mode='auto')

    tf_path = path_to_save + "Model/tf"
    fullModelSave = ExportModelCallback(tf_path, preprocessor.get_encoder(), lr)


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