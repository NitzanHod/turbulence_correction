import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


def float2int(img):
    """
    Convert the type of the np array fron float to uint8 - help for saving image with cv2
    :param img:
    :return:
    """
    assert img.dtype == np.float, "ERROR, dtype no valid in float2int"
    img = img * 255
    return img.astype('uint8')


def display_batch(gt, distorted_images):
    """
    Display a batch returned by the dataset
    :param gt: np array (h, w, 3)
    :param distorted_images: (n_frames, h, w, 3)
    :return:
    """
    n_distorted_images_to_display = min(distorted_images.shape[0], 3)

    # plot a sample of the distorted image in a figure
    fig, ax = plt.subplots(nrows=1,
                           ncols=n_distorted_images_to_display,
                           figsize=(8, 3))
    for idx in range(n_distorted_images_to_display):
        ax[idx].imshow(distorted_images[idx])
        ax[idx].set_title('Distorted Image no.' + str(idx + 1) + '/' + str(distorted_images.shape[0]))
    fig.tight_layout()

    # plot the original image on another figure
    plt.figure()
    plt.imshow(gt)
    plt.title('Original Image')
    plt.tight_layout()
    plt.show()


def create_folders():
    '''
    Function which creates a folders for saving files
    :return:
    '''
    mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir, exist_ok=True)

    # mydir_ = os.path.join(mydir, 'reconstruction')
    # os.makedirs(mydir_, exist_ok=True)
    #
    # mydir_ = os.path.join(mydir, 'learning_curve')
    # os.makedirs(mydir_, exist_ok=True)
    #
    # mydir_ = os.path.join(mydir, 'latent_space')
    # os.makedirs(mydir_, exist_ok=True)
    #
    # mydir_ = os.path.join(mydir, 'models')
    # os.makedirs(mydir_, exist_ok=True)
    return mydir
