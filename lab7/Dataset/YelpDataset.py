import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow_datasets as tfds
import tensorflow as tf

sys.path.append('config')
import settings
 
def download_data(path = settings.DATA_PATH, val_percent = settings.VAL_PERCENT):
    dataset = tfds.load('yelp_polarity_reviews', data_dir=path, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    
    train_size = len(train_dataset)
    val_size = int(train_size * val_percent)
    
    val_dataset = train_dataset.take(val_size)
    train_dataset = train_dataset.skip(val_size)
    
    return train_dataset, val_dataset, test_dataset

def create_train_pipeline(dataset, buffer_size = settings.BUFFER_SIZE, batch_size = settings.BATCH_SIZE):
    return dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_test_pipeline(dataset, batch_size = settings.BATCH_SIZE):
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_data(path = settings.DATA_PATH, val_percent = settings.VAL_PERCENT, buffer_size = settings.BUFFER_SIZE, batch_size = settings.BATCH_SIZE):
    train_ds, val_ds, test_ds = download_data(path, val_percent)
    
    train_ds = create_train_pipeline(train_ds, buffer_size, batch_size)
    val_ds = create_test_pipeline(val_ds, batch_size)
    test_ds = create_test_pipeline(test_ds, batch_size)
    
    return train_ds, val_ds, test_ds