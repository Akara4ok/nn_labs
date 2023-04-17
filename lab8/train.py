import sys
import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

sys.path.append('Dataset')
from dataset import LjSpeechDataset

sys.path.append('config')
import settings

sys.path.append('Model')
from model import Speech2Text
from loss_function import CTCLoss

sys.path.append('Preprocessing')
from preprocessing import LJSpeechPreprocessor

sys.path.append('Postprocessing')
from postprocessing import LJSpeechPostprocessing

sys.path.append('Callbacks')
from validation_callback import ValidationCallback

def train(version,
          last_checkpoint = None,
          data_path = settings.DATA_PATH,
          batch_size = settings.BATCH_SIZE,
          save_folder = settings.SAVE_FOLDER,
          epochs = settings.EPOCHS,
          lr = settings.DEFAULT_LR):
    
    speechDataset = LjSpeechDataset(
        data_path=data_path,
        charlist_file=settings.CHARLIST_PATH,
        batch_size=batch_size
    )
    
    charlist = speechDataset.get_charlist()
    wavs_path = speechDataset.get_wavs_path()
    
    preprocessor = LJSpeechPreprocessor(charlist, wavs_path)
    (train_ds, val_ds, test_ds) = speechDataset.create_data_pipelines(preprocessor)
    
    char_to_num = preprocessor.get_char_to_num()
    num_to_char = preprocessor.get_num_to_char()
    
    postprocessor = LJSpeechPostprocessing(num_to_char, charlist)
    
    model = Speech2Text(input_dim=settings.FFT_LENGTH // 2 + 1, output_dim=char_to_num.vocabulary_size(), rnn_units=512)
    
    model.compile(loss=CTCLoss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    
    path_to_save = save_folder  + '/' + version + '/'

    checkpoint_dir = path_to_save + "Checkpoints/"
    checkpoint_path = checkpoint_dir + "cp-{epoch:04d}.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                monitor='val_loss', 
                                verbose=1,
                                save_weights_only = True, 
                                mode='auto')
    
    init_epoch = 0
    if(last_checkpoint != None):
        checkpoint_name = checkpoint_dir + f"cp-{last_checkpoint:04d}.ckpt"
        init_epoch = last_checkpoint
        model.load_weights(checkpoint_name)

    tf_path = path_to_save + "Model/tf"
    fullModelSave = ModelCheckpoint(filepath=tf_path, 
                                monitor='val_loss', 
                                verbose=1,
                                save_best_only=True,
                                mode='auto')


    log_dir = path_to_save + "Logs/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    val_callback = ValidationCallback(val_ds, postprocessor)

    callbacks_list = [checkpoint, tensorboard_callback, fullModelSave, val_callback]

    model.fit(
        train_ds,
        epochs = epochs,
        initial_epoch=init_epoch,
        shuffle=False,
        validation_data = val_ds,
        callbacks = callbacks_list,
        verbose = 1)



if __name__ == '__main__':
    tf.random.set_seed(settings.RANDOM_SEED)

    parser=argparse.ArgumentParser()
    
    parser.add_argument("--version", "-v", default="v1", help="version of the model", type=str)
    parser.add_argument("--last-checkpoint", "-c", default=None, help="last checkpoint", type=int)
    
    parser.add_argument("--data-path", "-d", default=settings.DATA_PATH, help="path to dataset", type=str)
    parser.add_argument("--save-folder", "-s", default=settings.SAVE_FOLDER, help="path to save models and logs", type=str)
    
    
    parser.add_argument("--epochs", "-e", default=settings.EPOCHS, help="number of epochs", type=int)
    parser.add_argument("--batch-size", "-b", default=settings.BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--learning-rate", "-l", default=settings.DEFAULT_LR, help="learning rate", type=float)

    args = vars(parser.parse_args())
    

    train(args['version'],
          args['last_checkpoint'],
          args['data_path'],
          args['batch_size'],
          args['save_folder'],
          args['epochs'],
          args['learning_rate'])