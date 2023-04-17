import sys
import tensorflow as tf
import numpy as np
from spell_correction import SpellCorrection

sys.path.append('config')
import settings

class LJSpeechPostprocessing:
    def __init__(self, num_to_char, charlist, text_coorection_file = settings.SPELL_FILE_PATH) -> None:
        self.num_to_char = num_to_char
        self.spell_correction = SpellCorrection(text_coorection_file, charlist)
    
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(self.num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text
    
    def get_num_to_char(self):
        return self.num_to_char
    
    def postprocess(self, predictions, use_spell_correction = True):
        predictions = self.decode_batch_predictions(predictions)
        
        if use_spell_correction:
            for i, prediction in enumerate(predictions):
                words = prediction.split()
                words = [self.spell_correction.correction(word) for word in words]
                predictions[i] = ' '.join(words)
        
        return predictions
        