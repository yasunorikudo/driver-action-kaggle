import numpy as np
import cv2
import glob
from progressbar import ProgressBar

files = glob.glob('./data/train/*')
lists = []
for f in files:
    lists += glob.glob(f+'/*')
lists.sort()

mean = np.zeros((480, 640, 3), dtype=np.float64)
pbar = ProgressBar(len(lists))
for i, l in enumerate(lists):
    mean += cv2.imread(l).astype(np.float64) / len(lists)
    pbar.update(i+1)

np.save('./data/mean.npy', mean)
cv2.imwrite('./data/mean.jpg', mean)
