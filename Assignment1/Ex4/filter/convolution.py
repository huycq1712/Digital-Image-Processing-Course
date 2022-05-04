import numpy as np
from PIL import Image
import cv2

def conv(image ,kernel, padding, key_f=None):
    """Compute convolution

    Args:
        image (numpy.array - h,w,d): image read as numpy array
        kernel (numpy.array - h,w,d): kernel matrix
        padding (bool): use zero padding or not
        key_f (function object, optional): a function(args: local_maxtrix, kernel) is used to compute value for a position when kernel slide over it. Defaults to None.

    Returns:
        numpy.array: image as a numpy.array
    """
    
    if key_f is None:
        #element wise for regular convolution
        def element_wise(local_matrix, kernel):
            return np.sum((local_matrix*kernel), axis=(0,1))
        key_f = element_wise 
   
    height, width, depth =  image.shape
    k_height, k_width, k_depth = kernel.shape
    
    assert k_height < height and k_width < width, 'kernel size must be less than image'
    
    #padding
    if padding:
        p_height, p_width, p_depth = height+k_height-1, width+k_width-1, depth
        o_height, o_width, o_depth = height, width, depth
        padded_image = np.zeros((p_height, p_width, p_depth))
        padded_image[k_height//2:p_height-k_height//2, k_width//2:p_width-k_width//2] = image
        
    else:
        o_height, o_width, o_depth = height-k_height+1, width-k_width+1, depth
        p_height, p_width, p_depth = height, width, depth
        padded_image = image
    
    
    out_image = np.zeros((o_height, o_width, o_depth))
    
    for x in range(k_height//2, p_height-k_height//2):
        for y in range(k_width//2, p_width-k_width//2):
            out_image[x - k_height//2, y-k_height//2, :] = key_f(padded_image[x-k_height//2:x+k_height//2+1, y-k_width//2:y+k_width//2+1, :], kernel)
    
    return out_image