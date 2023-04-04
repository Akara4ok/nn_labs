import sys
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

sys.path.append('Datasets')
from StanfordDataset import StanfordDataset

sys.path.append('config')
import settings

sys.path.append('Preprocessings')
from StanfordPreprocessing import StanfordPreprocessing

import matplotlib.pyplot as plt

#from cli
stanfordDataset = StanfordDataset(
    data_path=settings.DATA_PATH,
    label_path=settings.LABELS_PATH,
    batch_size=settings.BATCH_SIZE
)
class_names = stanfordDataset.get_all_labels()

preprocessor = StanfordPreprocessing(class_names, 'basenji', settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH)
(train_ds, val_ds, test_ds) = stanfordDataset.create_data_pipelines(preprocessor)


