import sys
import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from jiwer import wer

sys.path.append('Dataset')
from dataset import LjSpeechDataset

sys.path.append('config')
import settings

sys.path.append('Model')
from model import Speech2Text
from loss_function import CTCLoss

sys.path.append('Preprocessing')
from preprocessing import LJSpeechPreprocessor

sys.path.append('Postprocessing')
from postprocessing import LJSpeechPostprocessing

sys.path.append('Callbacks')
from validation_callback import ValidationCallback

def train(version,
          data_path = settings.DATA_PATH,
          batch_size = settings.BATCH_SIZE,
          save_folder = settings.SAVE_FOLDER,
          use_corection = 1):
    
    speechDataset = LjSpeechDataset(
        data_path=data_path,
        charlist_file=settings.CHARLIST_PATH,
        batch_size=batch_size
    )
    
    charlist = speechDataset.get_charlist()
    wavs_path = speechDataset.get_wavs_path()
    
    preprocessor = LJSpeechPreprocessor(charlist, wavs_path)
    (train_ds, val_ds, test_ds) = speechDataset.create_data_pipelines(preprocessor)
    
    char_to_num = preprocessor.get_char_to_num()
    num_to_char = preprocessor.get_num_to_char()
    
    postprocessor = LJSpeechPostprocessing(num_to_char, charlist)
    
    tf_path = save_folder  + '/' + version + '/' + "Model/tf"
    model = tf.keras.models.load_model(tf_path, compile=False)

    i = 0

    predictions = []
    targets = []
    for batch in test_ds:
        X, y = batch
        batch_predictions = model.predict(X, verbose=0)
        batch_predictions = postprocessor.postprocess(batch_predictions, use_corection)
        predictions.extend(batch_predictions)
        for label in y:
            label = (
                tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            )
            targets.append(label)
    wer_score = wer(targets, predictions)
    print("-" * 100)
    print(f"Word Error Rate: {wer_score:.4f}")
    print("-" * 100)
    for i in range(100):
        print(f"Target    : {targets[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 100)


if __name__ == '__main__':
    tf.random.set_seed(settings.RANDOM_SEED)

    parser=argparse.ArgumentParser()
    
    parser.add_argument("--version", "-v", default="v1", help="version of the model", type=str)
    parser.add_argument("--data-path", "-d", default=settings.DATA_PATH, help="path to dataset", type=str)
    parser.add_argument("--save-folder", "-s", default=settings.SAVE_FOLDER, help="path to save models and logs", type=str)
    parser.add_argument("--batch-size", "-b", default=settings.BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--use-correction", "-c", default=1, help="use spell correction", type=int)

    args = vars(parser.parse_args())
    

    train(args['version'],
          args['data_path'],
          args['batch_size'],
          args['save_folder'],
          args['use_correction'])