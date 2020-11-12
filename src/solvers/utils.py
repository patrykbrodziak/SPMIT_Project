import tensorflow as tf


def slice_update(tensor, indices, updates):
    updated = tensor.numpy()
    updated[indices.numpy()] = updates

    return tf.convert_to_tensor(updated)
