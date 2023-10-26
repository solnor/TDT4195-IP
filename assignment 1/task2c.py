import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im, normalize
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "duck.jpeg"))
plt.imshow(im)


def convolve_im(im, kernel,
                ):
    """ A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    assert len(im.shape) == 3
    assert kernel.shape[0]%2 == 1
    assert kernel.shape[1]%2 == 1

    kernel   = np.flipud(np.fliplr(kernel))
    k_center = (kernel.shape[0]-1)//2

    out_im = np.copy(im)
    padded_im = np.pad(np.copy(im), ((k_center,k_center),(k_center,k_center),(0,0)), 'constant', constant_values=0)

    W = im.shape[0]
    H = im.shape[1]

    start_w = k_center
    stop_w  = W

    start_h = k_center
    stop_h  = H

    ones = np.ones((kernel.shape[0],))
    for w in range(start_w, stop_w):
        for h in range(start_h, stop_h):
            for c in range(0, 3):
                im_slice = padded_im[w-k_center:w+k_center+1,h-k_center:h+k_center+1,c]
                sum = np.dot(np.dot(im_slice*kernel,ones), ones)
                out_im[w-k_center,h-k_center,c] = sum
    return out_im


if __name__ == "__main__":
    # Define the convolutional kernels
    h_b = 1 / 256 * np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Convolve images
    im_smoothed = convolve_im(im.copy(), h_b)
    save_im(output_dir.joinpath("im_smoothed.jpg"), im_smoothed)
    im_sobel = convolve_im(im, sobel_x)
    save_im(output_dir.joinpath("im_sobel.jpg"), im_sobel)

    # DO NOT CHANGE. Checking that your function returns as expected
    assert isinstance(
        im_smoothed, np.ndarray),         f"Your convolve function has to return a np.array. " + f"Was: {type(im_smoothed)}"
    assert im_smoothed.shape == im.shape,         f"Expected smoothed im ({im_smoothed.shape}" + \
        f"to have same shape as im ({im.shape})"
    assert im_sobel.shape == im.shape,         f"Expected smoothed im ({im_sobel.shape}" + \
        f"to have same shape as im ({im.shape})"
    plt.subplot(1, 2, 1)
    plt.imshow(normalize(im_smoothed))

    plt.subplot(1, 2, 2)
    plt.imshow(normalize(im_sobel))
    plt.show()
