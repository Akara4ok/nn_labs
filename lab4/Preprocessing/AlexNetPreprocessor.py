import tensorflow as tf

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label

def create_data_pipeline(dataset, batch_size):
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    return dataset.shuffle(buffer_size = dataset_size).map(process_images).batch(batch_size = batch_size, drop_remainder = True)
    