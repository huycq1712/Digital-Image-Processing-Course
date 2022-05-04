import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from convolution import conv


def mean_filter(image, kernel_size):
    
    kernel = np.ones(shape=(kernel_size, kernel_size, image.shape[-1]))
    
    def mean(local_matrix, kernel):
        return np.sum((local_matrix*kernel), axis=(0,1))/kernel_size**2
    
    return conv(image, kernel, padding=True, key_f=mean)


def median_filter(image, kernel_size):
    
    kernel = np.random.randint(0, 1,size=(kernel_size,kernel_size,1))
    
    def median(local_matrix, kernel):
        local_array = local_matrix.copy().reshape((-1,local_matrix.shape[-1]))
        local_array = np.sort(local_array, axis=0)
        return local_array[len(local_array)//2,:]
    
    return conv(image, kernel, padding=True, key_f=median)


def gaussian_filter(image, kernel_size, sigma, constant):
    
    def gaussian_function(x, y, sigma, constant):
        return constant*np.exp(-(x**2+y**2)/(2*sigma**2))
    
    def compute_gaussian_matrix(n, sigma, constant):
        r = np.arange(-n//2+1,n//2+1)
        index_matrix = np.empty((n,n,2),dtype=int)
        index_matrix[:,:,0] = r[:,None]
        index_matrix[:,:,1] = r
       
        gaussian_function_vectorized = np.vectorize(gaussian_function)
        gaussian_matrix = gaussian_function_vectorized(index_matrix[:,:,0], index_matrix[:,:,1], sigma, constant)
        gaussian_matrix = gaussian_matrix/gaussian_function(-n//2+1, n//2, sigma, constant)
        
        gaussian_matrix = (gaussian_matrix/np.sum(gaussian_matrix))
        
        return gaussian_matrix
    
    kernel = compute_gaussian_matrix(kernel_size, sigma, constant)
    image = conv(image, kernel[:,:,None], padding=True)
    #image /= image.max()/255.0
    
    return image

def laplacian_filter(image, type, constant):
    
    def kernel_compute(type):
        if type == "1":
            kernel = - np.array([[1,1,1],
                       [1,-8,1],
                       [1,1,1]])[:,:,None]
        if type == "2":
            kernel = - np.array([[0,1,0],
                       [1,-4,1],
                       [0,1,0]])[:,:,None]
        return kernel
    
    def laplacian_compute(local_matrix, kernel, constant=constant):
        
        #h, w = local_maxtrix.shape[:2]
        return np.sum((local_matrix*kernel), axis=(0,1))
    
    image = image + constant*conv(image, kernel=kernel_compute(type), padding=True, key_f = laplacian_compute)
    image /= image.max()/255.0
    
    return image
        


if __name__ == '__main__':
    
    image = cv2.imread("/home/huycq1712/Code/School/Digital-Image-Processing-Course/Assignment1/Ex4/filter/Sharpen.png")
    #image = np.array(image)
    kernel = np.array([[1,1,1],
                       [1,-8,1],
                       [1,1,1]])[:,:,None]
    
    #kernel = np.random.randint(0, 1,size=(9,9,1))
    
    #image = conv(image, kernel, padding=True)
    #image = mean_filter(image, -1,kernel_size=5)
    image = laplacian_filter(image, type="1", constant=-1)
   
    
    
    #im = Image.fromarray(image.astype(int))
    #im.save("your_file.png")
    #image /= image.max()/255.0
    
    cv2.imwrite("mylaplasharp_3.png", image)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.imread("/home/huycq1712/Code/School/Digital-Image-Processing-Course/Assignment1/Ex4/filter/Sharpen.png")
    im = cv2.filter2D(im,-1 ,kernel)

    cv2.imwrite("opencvlaplasharp.png", image)
    #imss = [image.astype(int), im.astype(int)]
    
    
    #img = cv2.imread('/home/huycq1712/Code/School/DigitalImageProcessing/Week1/filter/Sharpen.png')
    
