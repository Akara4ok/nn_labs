import tensorflow as tf
import sys

sys.path.append('config')
import settings

class LJSpeechPreprocessor:
    def __init__(self, charlist, wavs_path) -> None:
        self.charlist = charlist
        self.wavs_path = wavs_path
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=self.charlist, oov_token="")
        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True
        )
    
    def get_char_to_num(self):
        return self.char_to_num
    
    def get_num_to_char(self):
        return self.num_to_char
    
    def encode_single_sample(self, wav_file, label):
        #read and decode audio
        file = tf.io.read_file(self.wavs_path + wav_file + ".wav")
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        
        #create and normalize spectrogram
        spectrogram = tf.signal.stft(
            audio, frame_length=settings.FRAME_LENGTH, frame_step=settings.FRAME_STEP, fft_length=settings.FFT_LENGTH
        )
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)

        #process labels
        label = tf.strings.lower(label)
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        label = self.char_to_num(label)
        
        return spectrogram, label