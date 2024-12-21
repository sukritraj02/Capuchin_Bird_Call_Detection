import tensorflow as tf

def create_tf_dataset(features, labels, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))
    dataset = dataset.batch(batch_size)
    return dataset
