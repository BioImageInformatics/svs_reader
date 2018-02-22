import cv2
import numpy as np
import sys

sys.path.insert(0, '..')
from slide import Slide

print '\nSlide at 20x'
s = Slide(slide_path='/home/nathan/data/ccrcc/TCGA_KIRC/TCGA-A3-3346-01Z-00-DX1.95280216-fd71-4a03-b452-6e3d667f2542.svs',
        process_mag=20, )

print '\nSlide at 10x'
s = Slide(slide_path='/home/nathan/data/ccrcc/TCGA_KIRC/TCGA-A3-3346-01Z-00-DX1.95280216-fd71-4a03-b452-6e3d667f2542.svs',
        process_mag=10, )

print '\nSlide at 5x'
s = Slide(slide_path='/home/nathan/data/ccrcc/TCGA_KIRC/TCGA-A3-3346-01Z-00-DX1.95280216-fd71-4a03-b452-6e3d667f2542.svs',
        process_mag=5, )

s.print_info()

for idx, img in enumerate(s.generator()):
    # print img
    cv2.imwrite('debug/{}.jpg'.format(idx), img[:,:,::-1])
