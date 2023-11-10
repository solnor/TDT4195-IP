import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils


def convolve_im(im: np.array,
                fft_kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the frequency kernel (fft_kernel),
        and returns the resulting image.

        "verbose" can be used for turning on/off visualization
        convolution

    Args:
        im: np.array of shape [H, W]
        fft_kernel: np.array of shape [H, W] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    
    # Fourier transform image, shift the frequencies to centre
    im_fft = np.fft.fft2(im)
    im_fft = np.fft.fftshift(im_fft)
    
    # Shift the frequencies of the kernel
    fft_kernel = np.fft.fftshift(fft_kernel)


    # Convolve the image with the kernel in the frequency domain
    conv_result_fft = np.multiply(im_fft, fft_kernel)
    # Shift the frequencies back before inverse Fourier transforming
    conv_result_fft = np.fft.fftshift(conv_result_fft)
    conv_result = np.fft.ifft2(conv_result_fft).real
    # Shift frequencies s.t. zero-frequency is in centre again
    conv_result_fft = np.fft.fftshift(conv_result_fft)

    # Get magnitude of images in frequency domain for visualisation. 
    im_fft          = np.log(np.abs(im_fft) + 1)
    fft_kernel      = np.log(np.abs(fft_kernel) + 1)
    conv_result_fft = np.log(np.abs(conv_result_fft) + 1)

    if verbose:
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")

        plt.subplot(1, 5, 2)
        # Visualize FFT
        plt.imshow(im_fft, cmap="gray")
        
        plt.subplot(1, 5, 3)
        # Visualize FFT kernel
        plt.imshow(fft_kernel, cmap="gray")
        
        plt.subplot(1, 5, 4)
        # Visualize filtered FFT image
        plt.imshow(conv_result_fft, cmap="gray")
        
        plt.subplot(1, 5, 5)
        # Visualize filtered spatial image
        plt.imshow(conv_result, cmap="gray")
    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True
    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)
    # DO NOT CHANGE
    frequency_kernel_low_pass = utils.create_low_pass_frequency_kernel(
        im, radius=50)
    image_low_pass = convolve_im(im, frequency_kernel_low_pass,
                                 verbose=verbose)
    # DO NOT CHANGE
    frequency_kernel_high_pass = utils.create_high_pass_frequency_kernel(
        im, radius=50)
    image_high_pass = convolve_im(im, frequency_kernel_high_pass,
                                  verbose=verbose)
    
    if verbose:
        plt.show()
    utils.save_im("camera_low_pass.png", image_low_pass)
    utils.save_im("camera_high_pass.png", image_high_pass)

    

