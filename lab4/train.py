import sys
import argparse
import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from keras.callbacks import ModelCheckpoint

sys.path.append('Models')
from AlexNet import AlexNet

sys.path.append('config')
import settings

sys.path.append('Preprocessing')
from AlexNetPreprocessor import create_data_pipeline 

def train(version,
          validation_num = settings.VALIDATION_NUM,
          save_folder = settings.SAVE_FOLDER,
          epochs = settings.EPOCHS,
          batch_size = settings.BATCH_SIZE,
          lr = settings.DEFAULT_LR):
    
    #load data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
    validation_images, validation_labels = train_images[:validation_num], train_labels[:validation_num]
    train_images, train_labels = train_images[validation_num:], train_labels[validation_num:]
    
    #creating tf.data
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))


    #creating data pipelines
    train_ds = create_data_pipeline(train_ds, batch_size) 
    validation_ds = create_data_pipeline(validation_ds, batch_size)

    model = AlexNet()

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

    model.compile(loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'], 
                  optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))
    
    path_to_save = save_folder + '/' + version + '/'

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
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_list = [checkpoint, tensorboard_callback, fullModelSave]

    model.fit(
        train_ds,
        epochs = epochs, 
        validation_data = validation_ds,
        callbacks = callbacks_list,
        verbose = 1)


if __name__ == '__main__':
    tf.random.set_seed(settings.RANDOM_SEED)

    parser=argparse.ArgumentParser()

    parser.add_argument("--version", "-v", help="version of the model", type=str)
    
    parser.add_argument("--validation-num", default=settings.VALIDATION_NUM, help="number of validation samples", type=int)

    parser.add_argument("--epochs", "-e", default=settings.EPOCHS, help="number of epochs", type=int)
    parser.add_argument("--batch-size", "-b", default=settings.BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--learning-rate", "-l", default=settings.DEFAULT_LR, help="learning rate", type=float)

    parser.add_argument("--save-folder", "-s", default=settings.SAVE_FOLDER, help="path to save output", type=str)

    args = vars(parser.parse_args())
    

    train(args['version'],
          args['validation_num'],
          args['save_folder'],
          args['epochs'],
          args['batch_size'],
          args['learning_rate'])