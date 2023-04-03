import tensorflow as tf
import pathlib
import numpy as np
import sys
sys.path.append('Preprocessings')
from StanfordPreprocessing import StanfordPreprocessing
sys.path.append('config')
import settings

class StanfordDataset:
    def __init__(self, 
                 data_path, 
                 label_path,
                 desired_value,
                 batch_size) -> None:
        
        self.data_path = data_path
        self.label_path = label_path
        self.random_seed = settings.RANDOM_SEED
        self.val_percent = settings.VAL_PERCENT
        self.test_percent = settings.TEST_PERCENT
        self.labels = self.read_labels(label_path)
        self.desired_value = desired_value
        self.preprocessor = StanfordPreprocessing(self.labels, self.desired_value, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH)
        self.batch_size = batch_size
        

    def load_data(self, path, random_seed, val_percent, test_percent):
        data_dir = pathlib.Path(path)
        data_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=True, seed=random_seed)
        image_count = len(data_ds)
        val_size = int(image_count * val_percent)
        test_size = int(image_count * test_percent)
        
        train_ds = data_ds.skip(test_size + val_size)
        val_ds = data_ds.skip(test_size).take(val_size)
        test_ds = data_ds.take(test_size)
        
        return (train_ds, val_ds, test_ds)

    def create_labels(data_path, save_path):
        data_dir = pathlib.Path(data_path)
        class_names = sorted([item.name for item in data_dir.glob('*')])
        file_str = '\n'.join(class_names)
        with open(save_path, "w") as file:
            file.write(file_str)

    
    def read_labels(self, path):
        with open(path, 'r') as file:
            data = file.read()
        
        return np.array(data.split('\n'))

    def create_train_pipeline(self, ds):
        image_count = len(ds)
        ds = ds.shuffle(buffer_size = image_count, reshuffle_each_iteration=True).map(self.preprocessor.process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size = self.batch_size)
        return ds

    def create_test_pipeline(self, ds):
        ds = ds.map(self.preprocessor.process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size = self.batch_size)
        return ds
    
    def get_datasets(self):
        (train_ds, val_ds, test_ds) = self.load_data(self.data_path, self.random_seed, self.val_percent, self.test_percent)
        train_ds = self.create_train_pipeline(train_ds)
        val_ds = self.create_test_pipeline(val_ds)
        test_ds = self.create_test_pipeline(test_ds)
        return (train_ds, val_ds, test_ds)
    
    def get_all_labels(self):
        return self.labels
    
    def get_desired_label(self):
        return self.desired_value