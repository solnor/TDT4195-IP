import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "duck.jpeg"))
plt.imshow(im)
plt.show()


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    
    W  = im.shape[0]
    H  = im.shape[1]
    CV = im.shape[2]

    im_copy = np.copy(im)

    for w in range(W):
        R = np.reshape( 0.212*im_copy[w,:,0], (H, 1))
        G = np.reshape(0.7152*im_copy[w,:,1], (H, 1))
        B = np.reshape(0.0722*im_copy[w,:,2], (H, 1))

        grey = np.concatenate((R,G,B), axis = 1)
        grey = np.reshape(grey.sum(axis=1), (grey.shape[0], 1))
        grey = np.concatenate((grey,grey,grey), axis=1)
        im_copy[w,:,:] = grey
    return im_copy


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("duck_greyscale.jpeg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")

plt.show()

def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # YOUR CODE HERE
    W  = im.shape[0]
    H  = im.shape[1]
    CV = im.shape[2]

    im_copy = np.copy(im)

    for w in range(W):
        R = np.reshape(np.subtract(np.ones((H,)), im_copy[w,:,0]), (H, 1))
        G = np.reshape(np.subtract(np.ones((H,)), im_copy[w,:,1]), (H, 1))
        B = np.reshape(np.subtract(np.ones((H,)), im_copy[w,:,2]), (H, 1))
        rgb_inverse = np.concatenate((R,G,B), axis=1)
        im_copy[w,:,:] = rgb_inverse
    return im_copy

im_inverse = inverse(im)
save_im(output_dir.joinpath("duck_inverse.jpeg"), im_inverse, cmap=None)
plt.imshow(im_inverse)

plt.show()