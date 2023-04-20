import os
import numpy as np
import tensorflow as tf

class StanfordPreprocessing:
    def __init__(self, labels, img_height, img_width) -> None:
        self.labels = labels
        self.img_height = img_height
        self.img_width = img_width

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.labels
        return tf.argmax(one_hot)

    def decode_img(self, img):
        img = tf.io.decode_jpeg(img, channels=3)
        return tf.image.resize(img, [self.img_height, self.img_width])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label
    
    def process_single_img(self, file_path):
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img

