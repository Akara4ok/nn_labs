import sys
import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

sys.path.append('Dataset')
from dataset import Dataset

sys.path.append('config')
import settings

sys.path.append('Preprocessing')
from preprocessing import LogoPreprocessing

import matplotlib.pyplot as plt

def train(version,
          data_path = settings.DATA_PATH,
          label_path = settings.LABELS_PATH,          
          batch_size = settings.BATCH_SIZE,
          save_folder = settings.SAVE_FOLDER,
          epochs = settings.EPOCHS,
          lr = settings.DEFAULT_LR):
    
    logoDataset = Dataset(
        data_path=data_path,
        label_path=label_path,
        batch_size=batch_size
    )
    
    class_names = logoDataset.get_all_labels()
    
    preprocessor = LogoPreprocessing(class_names, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH)
    (train, val, test) = logoDataset.create_data_pipelines(preprocessor)


if __name__ == '__main__':
    tf.random.set_seed(settings.RANDOM_SEED)

    parser=argparse.ArgumentParser()
    
    parser.add_argument("--version", "-v", default="v1", help="version of the model", type=str)
    
    parser.add_argument("--data-path", "-d", default=settings.DATA_PATH, help="path to dataset", type=str)
    parser.add_argument("--labels-path", default=settings.LABELS_PATH, help="path to labels", type=str)
    parser.add_argument("--save-folder", "-s", default=settings.SAVE_FOLDER, help="path to save models and logs", type=str)
    
    
    parser.add_argument("--epochs", "-e", default=settings.EPOCHS, help="number of epochs", type=int)
    parser.add_argument("--batch-size", "-b", default=settings.BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--learning-rate", "-l", default=settings.DEFAULT_LR, help="learning rate", type=float)

    args = vars(parser.parse_args())
    

    train(args['version'],
          args['data_path'],
          args['labels_path'],
          args['batch_size'],
          args['save_folder'],
          args['epochs'],
          args['learning_rate'])
