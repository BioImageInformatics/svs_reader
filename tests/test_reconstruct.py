import cv2
import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.insert(0, '..')
from slide import Slide

print '\nSlide at 5x'
s = Slide(slide_path='/home/nathan/data/ccrcc/TCGA_KIRC/TCGA-A3-3346-01Z-00-DX1.95280216-fd71-4a03-b452-6e3d667f2542.svs',
        process_mag=5,
        oversample_factor=2 )
s.initialize_output(n_classes=3)
s.print_info()

for idx, img in enumerate(s.generator()):
    s.place(img[:,:,::-1], idx)

reconstruction = s.output_img
print reconstruction.shape

plt.imshow(reconstruction)
plt.show()
