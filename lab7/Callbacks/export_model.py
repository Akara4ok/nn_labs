import os
import tensorflow as tf
        
class ExportModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, tf_path, encoder, lr):
        super().__init__()
        self.tf_path = tf_path
        self.encoder = encoder
        self.lr = lr
        self.best_val = float('inf')

    def on_epoch_end(self, epoch: int, logs=None):
        if not os.path.exists(os.path.dirname(self.tf_path)):
            os.makedirs(os.path.dirname(self.tf_path))
        previous_loss = self.best_val
        if logs.get('val_loss') < self.best_val:
            self.best_val = logs.get('val_loss')
            
            export_model = tf.keras.Sequential([
                self.encoder, 
                self.model])

            export_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            metrics=['accuracy'], 
                            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
            
            export_model.save(self.tf_path)
            
            print("Epoch {}: val_loss improved from {:.5} to {:.5}, saving model to {}".format(epoch + 1, previous_loss, self.best_val, self.tf_path))