"""
Script which contains the function DistortBlur for simulating turbulence on an clean image.
written by oren, last modification: 25/03/2020
"""
import numpy as np
import cv2
from skimage.transform import PiecewiseAffineTransform, warp
import matplotlib.pyplot as plt


def DistortBlur(img, S, sigma_kernel_vertor_field, sigma_blur_image, N=15, M_distortion=1000, M_blur=50):
    """
        Return an artificially distorted image.
        :param img: np array - image=(h, w, 3)
        :param S: float - distortion strength
        :param sigma_kernel_vertor_field: float - std of the kernel when smoothing the vector field a each iteration
        :param N: int - half size of the patch
        :param M_distortion: int - number of iterations for generating vector field patches
        :param M_blur: int - number of iterations for performing high blur in random patches
        :return:
        """
    assert (N % 2) == 1, "N must be odd!."
    img_height, img_width, img_channel = img.shape
    # generate the grid of the src image
    src_cols = np.arange(0, img_width)
    src_rows = np.arange(0, img_height)
    src_cols, src_rows = np.meshgrid(src_cols, src_rows)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # generate the vector field
    vector_field = np.zeros(shape=(img_height, img_width, 2), dtype=np.float32)
    for i in range(M_distortion):
        x = np.random.randint(low=0, high=img_width - 2 * N) + N
        y = np.random.randint(low=0, high=img_height - 2 * N) + N
        vector_field_current_patch_x = np.random.randn(2 * N, 2 * N)
        vector_field_current_patch_y = np.random.randn(2 * N, 2 * N)
        vector_field_current_patch_x = S * cv2.GaussianBlur(vector_field_current_patch_x,
                                                            ksize=(N // 2, N // 2),
                                                            sigmaX=sigma_kernel_vertor_field)
        vector_field_current_patch_y = S * cv2.GaussianBlur(vector_field_current_patch_y,
                                                            ksize=(N // 2, N // 2),
                                                            sigmaX=sigma_kernel_vertor_field)
        vector_field[y - N:y + N, x - N:x + N, 0] = vector_field[y - N:y + N, x - N:x + N,
                                                    0] + vector_field_current_patch_x
        vector_field[y - N:y + N, x - N:x + N, 1] = vector_field[y - N:y + N, x - N:x + N,
                                                    1] + vector_field_current_patch_y
    vector_field[:, :, 0] = cv2.GaussianBlur(vector_field[:, :, 0],
                                             ksize=(N // 2, N // 2),
                                             sigmaX=sigma_kernel_vertor_field)
    vector_field[:, :, 1] = cv2.GaussianBlur(vector_field[:, :, 1],
                                             ksize=(N // 2, N // 2),
                                             sigmaX=sigma_kernel_vertor_field)
    # generate the grid of the ouput image
    dst_cols = np.arange(0, img_width)
    dst_rows = np.arange(0, img_height)
    dst_cols, dst_rows = np.meshgrid(dst_cols, dst_rows)
    dst_cols, dst_rows = dst_cols.astype('float32'), dst_rows.astype('float32')
    dst_rows += vector_field[:, :, 0]
    dst_cols += vector_field[:, :, 1]
    dst = np.dstack([dst_cols.flat, dst_rows.flat])[0]

    # compute the transform
    tform = PiecewiseAffineTransform()
    step = 20
    src, dst = src[step // 2:-1:step], dst[step // 2:-1:step]
    tform.estimate(src, dst)

    # perform the transform
    distorded_image = warp(img, tform, output_shape=(img_height, img_width))

    # blur the image globally
    distorded_image = cv2.GaussianBlur(distorded_image, (N, N), sigma_blur_image)

    # blur the image in random patch
    for i in range(M_blur):
        x = np.random.randint(low=0, high=img_width - 2 * N) + N
        y = np.random.randint(low=0, high=img_height - 2 * N) + N
        current_patch = distorded_image[y - N:y + N, x - N:x + N, :]
        current_patch = cv2.GaussianBlur(current_patch, (N, N), sigma_blur_image)
        distorded_image[y - (N - 1) // 2:y + (N - 1) // 2 + 1,
        x - (N - 1) // 2:x + (N - 1) // 2 + 1, :] = current_patch[(N - 1) // 2 + 1:3 * (N - 1) // 2 + 2,
                                                    (N - 1) // 2 + 1:3 * (N - 1) // 2 + 2, :]

    # blur the image globally
    distorded_image = cv2.GaussianBlur(distorded_image, (N, N), sigma_blur_image / 10)

    return distorded_image


if __name__ == '__main__':
    image_input = cv2.imread('../image_test_2.jpg') / 255.0
    image_input = cv2.resize(image_input, (420, 280))
    distorted_image = DistortBlur(image_input, S=1.0,
                                  sigma_kernel_vertor_field=0.7,
                                  sigma_blur_image=np.random.uniform(0.5, 1.5),
                                  N=15,
                                  M_distortion=1000,
                                  M_blur=50)

    # display results
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(8, 3),
                                   sharex=True, sharey=True)

    ax1.imshow(image_input[:, :, ::-1])
    ax1.set_title('input image')

    ax2.imshow(distorted_image[:, :, ::-1])
    ax2.set_title('distorted image')
    plt.tight_layout()
    plt.show()
