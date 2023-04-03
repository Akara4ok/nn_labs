import os
import numpy as np
import tensorflow as tf

class StanfordPreprocessing:
    def __init__(self, labels, desired_value, img_height, img_width) -> None:
        self.label = self.get_label(labels, desired_value)
        self.img_height = img_height
        self.img_width = img_width
        
    def get_label(self, labels, desired_value):
        for element in labels:
            if element.find(desired_value) >= 0:
                return element
        return 'Unknown'

    def get_one_hot_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return tf.cast(parts[-2] == self.label, tf.int32)

    def decode_img(self, img):
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(img, [self.img_height, self.img_width])

    def process_path(self, file_path):
        label = self.get_one_hot_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

