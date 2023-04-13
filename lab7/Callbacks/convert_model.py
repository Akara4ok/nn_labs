import os
import tensorflow as tf
        
class ConvertModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, tf_path, export_path, encoder, lr, epoch_freq):
        super().__init__()
        self.tf_path = tf_path
        self.export_path = export_path
        self.encoder = encoder
        self.lr = lr
        self.previous_bast_val = float('inf')
        self.best_val = float('inf')
        self.epoch_freq = epoch_freq
        self.threshold = 0.0001

    def on_epoch_end(self, epoch: int, logs=None):
        if not os.path.exists(os.path.dirname(self.export_path)):
            os.makedirs(os.path.dirname(self.export_path))
        
        if(logs.get('val_loss') < self.best_val):
            self.best_val = logs.get('val_loss')
            
        if (epoch % self.epoch_freq == 0):
            if(abs(self.previous_bast_val - self.best_val) < self.threshold):
                print("Epoch {}: val_loss did not improve from {:.5}".format(epoch + 1, self.previous_bast_val))
                return
            
            print("Epoch {}: val_loss improved from {:.5} to {:.5}, exporting model to {}".format(epoch + 1, self.previous_bast_val, self.best_val, self.export_path))
            
            export_model = tf.keras.Sequential([
                self.encoder, 
                self.model])

            export_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=['accuracy'], 
                            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
            
            export_model.save(self.export_path)
            