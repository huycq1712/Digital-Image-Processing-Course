import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('Contract.png')
def compute_hist(img):
    hist = np.zeros((256,), np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] += 1
    return hist

def equal_hist(hist):
    cumulator = np.zeros_like(hist, np.float64)
    for i in range(len(cumulator)):
        cumulator[i] = hist[:i].sum()
    print(cumulator)
    new_hist = (cumulator - cumulator.min()) / (cumulator.max() - cumulator.min()) * 255
    new_hist = np.uint8(new_hist)
    return new_hist

hist = compute_hist(img).ravel()
new_hist = equal_hist(hist)

h, w = img.shape[:2]
for i in range(h):
    for j in range(w):
        img[i, j] = new_hist[img[i, j]]

fig = plt.figure()
ax = plt.subplot(121)
plt.imshow(img, cmap='gray')

plt.subplot(122)
plt.plot(new_hist)
plt.show()


# Dùng OPEN CV
img = cv2.imread('Contract.png', 0)
img_equal_hist = cv2.equalizeHist(img)

fig, axes = plt.subplots(2, 2, figsize=(30, 20))
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Origin')
axes[0, 1].imshow(cv2.cvtColor(img_equal_hist, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Hist equal')
axes[1, 0].hist(img.flatten(), 256, [0, 256])
axes[1, 1].hist(img_equal_hist.flatten(), 256, [0, 256])
plt.show()
