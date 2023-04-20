import sys
import tensorflow as tf

sys.path.append('config')
import settings

class YelpPreprocessing():
    def __init__(self, train_dataset) -> None:
        self.encoder = self.create_encoder(train_dataset)

    def create_encoder(self, dataset):
        encoder = tf.keras.layers.TextVectorization(
            max_tokens=settings.VOCAB_SIZE)
        encoder.adapt(dataset.map(lambda text, label: text))
        return encoder

    def vectorize_text(self, text, label):
        text = tf.expand_dims(text, -1)
        return self.encoder(text), label
    
    def get_encoder(self):
        return self.encoder