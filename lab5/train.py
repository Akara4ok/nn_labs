import sys
import argparse
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

sys.path.append('Datasets')
from StanfordDataset import StanfordDataset

sys.path.append('config')
import settings

sys.path.append('Preprocessings')
from StanfordPreprocessing import StanfordPreprocessing

sys.path.append('Models')
from InceptionV3 import InceptionV3



def train(desired_value,
          version,
          data_path = settings.DATA_PATH,
          label_path = settings.LABELS_PATH,          
          batch_size = settings.BATCH_SIZE,
          save_folder = settings.SAVE_FOLDER,
          epochs = settings.EPOCHS,
          lr = settings.DEFAULT_LR):
    
    #from cli
    stanfordDataset = StanfordDataset(
        data_path=data_path,
        label_path=label_path,
        batch_size=batch_size
    )
    
    class_names = stanfordDataset.get_all_labels()

    preprocessor = StanfordPreprocessing(class_names, desired_value, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH)
    (train_ds, val_ds, test_ds) = stanfordDataset.create_data_pipelines(preprocessor)

    model = InceptionV3((settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3), 1)
    
    #values for schedules
    initial_learning_rate = 10**(-2)
    final_learning_rate = 10**(-4)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
    steps_per_epoch = len(train_ds)
    
    learning_rate = lr
    if(lr == -1):
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=initial_learning_rate,
                        decay_steps=steps_per_epoch,
                        decay_rate=learning_rate_decay_factor
                    )


    model.compile(loss='binary_crossentropy', 
                  metrics=['accuracy'], 
                  optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))
    
    
    
    path_to_save = save_folder + '/' + desired_value + '/' + version + '/'

    checkpoint_dir = path_to_save + "Checkpoints/"
    checkpoint_path = checkpoint_dir + "cp-{epoch:04d}.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                                monitor='val_loss', 
                                verbose=1,
                                save_weights_only = True, 
                                mode='auto')

    tf_path = path_to_save + "Model/tf"
    fullModelSave = ModelCheckpoint(filepath=tf_path, 
                                monitor='val_loss', 
                                verbose=1,
                                save_best_only=True,
                                mode='auto')


    log_dir = path_to_save + "Logs/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callbacks_list = [checkpoint, tensorboard_callback, fullModelSave]

    model.fit(
        train_ds,
        epochs = epochs, 
        shuffle=False,
        validation_data = val_ds,
        callbacks = callbacks_list,
        verbose = 1)
    

if __name__ == '__main__':
    tf.random.set_seed(settings.RANDOM_SEED)

    parser=argparse.ArgumentParser()

    parser.add_argument("--desired-value", help="name of dog", type=str)
    parser.add_argument("--version", "-v", help="version of the model", type=str)
    
    parser.add_argument("--data-path", "-d", default=settings.DATA_PATH, help="number of validation samples", type=str)
    parser.add_argument("--labels-path", default=settings.LABELS_PATH, help="number of validation samples", type=str)
    parser.add_argument("--save-folder", "-s", default=settings.SAVE_FOLDER, help="number of validation samples", type=str)
    
    
    parser.add_argument("--epochs", "-e", default=settings.EPOCHS, help="number of epochs", type=int)
    parser.add_argument("--batch-size", "-b", default=settings.BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--learning-rate", "-l", default=settings.DEFAULT_LR, help="learning rate", type=float)

    args = vars(parser.parse_args())
    

    train(args['desired_value'],
          args['version'],
          args['data_path'],
          args['labels_path'],
          args['batch_size'],
          args['save_folder'],
          args['epochs'],
          args['learning_rate'])