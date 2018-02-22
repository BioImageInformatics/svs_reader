import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import sys
import cv2

sys.path.insert(0, '..')
from slide import Slide

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tfe.enable_eager_execution(config=config)

print '\nSlide at 5x'
s = Slide(slide_path='/home/nathan/data/ccrcc/TCGA_KIRC/TCGA-A3-3346-01Z-00-DX1.95280216-fd71-4a03-b452-6e3d667f2542.svs',
        process_mag=5, )
s.print_info()

ds = tf.data.Dataset.from_generator(generator=s.generator, output_types=tf.float32)
ds = tfe.Iterator(ds)

for idx, x in enumerate(ds):
    print x.shape
    cv2.imwrite('debug/{}.jpg'.format(idx), x.numpy()[:,:,::-1])
