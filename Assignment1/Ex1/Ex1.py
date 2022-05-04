import cv2 as cv
import numpy as np

img = cv.imread(r'Lena.png')

cv.imshow('Gray Scale Image', img)

# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
cv.waitKey(0)

# cv2.destroyAllWindows() simply destroys all the windows we created.
cv.destroyAllWindows()

# The function cv2.imwrite() is used to write an image.
cv.imwrite('grayscale.jpg',img)





