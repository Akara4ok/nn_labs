import pandas as pd
import numpy as np
import tensorflow as tf
import sys

sys.path.append('config')
import settings

class LjSpeechDataset:
    def __init__(self, 
                 data_path = settings.DATA_PATH,
                 charlist_file = settings.CHARLIST_PATH,
                 batch_size = settings.BATCH_SIZE,
                 val_percent = settings.VAL_PERCENT, 
                 test_percent = settings.TEST_PERCENT) -> None:
        self.wavs_path = data_path + "/wavs/"
        metadata_path = data_path + "/metadata.csv"
        self.metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
        self.metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
        self.metadata_df = self.metadata_df[["file_name", "normalized_transcription"]]
        self.metadata_df = self.metadata_df.sample(frac=1, random_state = settings.RANDOM_SEED).reset_index(drop=True)
        
        self.charlist = self.read_charlist(charlist_file)
        self.batch_size = batch_size
        
        test_split = int(len(self.metadata_df) * test_percent)
        val_split = int(len(self.metadata_df) * (val_percent + test_percent))
        self.test_pd = self.metadata_df[:test_split]
        test_pd = self.metadata_df[:test_split]
        val_pd = self.metadata_df[test_split:val_split]
        train_pd = self.metadata_df[val_split:]
        
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (list(train_pd["file_name"]), list(train_pd["normalized_transcription"]))
        )
        self.val_ds = tf.data.Dataset.from_tensor_slices(
            (list(val_pd["file_name"]), list(val_pd["normalized_transcription"]))
        )
        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (list(test_pd["file_name"]), list(test_pd["normalized_transcription"]))
        )

        
    def read_charlist(self, path):
        char_list_file = open(path)
        line = char_list_file.readline()
        charlist = [x for x in line]
        return charlist
    
    def get_charlist(self):
        return self.charlist
    
    def get_wavs_path(self):
        return self.wavs_path
    
    def get_test_paths(self):
        return self.test_pd
    
    def create_data_pipeline(self, ds, preprocessor):
        ds = (
            ds.map(preprocessor.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(self.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return ds

    def create_data_pipelines(self, preprocessor):
        train_ds = self.create_data_pipeline(self.train_ds, preprocessor)
        val_ds = self.create_data_pipeline(self.val_ds, preprocessor)
        test_ds = self.create_data_pipeline(self.test_ds, preprocessor)
        return (train_ds, val_ds, test_ds)