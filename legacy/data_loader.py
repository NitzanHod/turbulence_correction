"""
Script containing the function returning the tensorflow dataset object.
"""
import tensorflow as tf
from glob import glob
import config
from legacy.utils import *


def decode_img(img_path):
    """
    Function for loading an image from a tf string
    :param img_path: img_path
    :return: image as tf tensor
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # pad the image so its shape is a multiple of 2**n_down_sampling - need for U_NET down-sample
    height, width = tf.cast(tf.shape(img)[0], dtype=tf.float32), tf.cast(tf.shape(img)[1], dtype=tf.float32)
    target_height = tf.math.ceil(height / 2 ** config.n_down_sampling) * 2 ** config.n_down_sampling
    target_width = tf.math.ceil(width / 2 ** config.n_down_sampling) * 2 ** config.n_down_sampling
    img = tf.image.pad_to_bounding_box(img, 0, 0, tf.cast(target_height, tf.int32), tf.cast(target_width, tf.int32))

    return img


def return_images_from_scene(distorted_list_path, gt_path):
    """
    Function that return a stack of distorted images of a single scene and the correspondent ground truth.
    It returns two tensors: one consisting of the distorted images (shape=(n_frames, h, w, 3)
    and the second one is the gt_image (shape=(h, w, 3)
    :param distorted_list_path: Tensor of strings
    :param gt_path: Tensor of strings
    """
    # select n_frames randomly from the list of distorted images
    distorted_list_path_shuffled = tf.random.shuffle(distorted_list_path)

    # read the GT image
    gt_image = decode_img(gt_path)

    # read n_frames distorted images
    distorted_images_tensor = []
    for idx in range(config.n_frames):
        distorted_images_tensor.append(decode_img(distorted_list_path_shuffled[idx]))
    distorted_images_tensor = tf.stack(distorted_images_tensor)

    return distorted_images_tensor, gt_image


def create_dataset(data_main_path_directory):
    """
    Return the tensorflow dataset.
    :param data_main_path_directory: path of the main directory folder containing the generated images
    """
    # initialize lists - each element in the list correspond to a single scene
    GT_list_path, distorted_list_path = [], []

    # update the list by getting all the images path
    scenes_path_list = glob(data_main_path_directory + '/*')
    for current_scene_path in scenes_path_list:
        current_GT_path = glob(current_scene_path + '/GT/*.jpg')[0]
        current_distorted_path = glob(current_scene_path + '/distorted/*.jpg')

        for _ in range(config.batch_size):
            # repeat the elements of the dataset in order to allow batch
            # (in a batch, all the images should be of the same shape,
            # so we can not mix between scenes)
            GT_list_path.append(current_GT_path)
            distorted_list_path.append(current_distorted_path)

    dataset = tf.data.Dataset.from_tensor_slices((distorted_list_path, GT_list_path))
    dataset = dataset.map(return_images_from_scene)
    dataset = dataset.batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == '__main__':
    dataset = create_dataset(data_main_path_directory=config.data_main_path_directory)

    for distorted_images, gt_image in dataset:
        distorted_images = distorted_images.numpy()
        gt_image = gt_image.numpy()
        display_batch(gt_image[0], distorted_images[0])
        a = 1
