import tensorflow as tf
import numpy as np
from jiwer import wer

class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, postprocessor):
        super().__init__()
        self.dataset = dataset
        self.postprocessor = postprocessor
        self.num_to_char = postprocessor.get_num_to_char()

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model.predict(X)
            batch_predictions = self.postprocessor.decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(self.num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)
