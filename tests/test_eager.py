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

print '\nslide at 5x'
slide_path = '/home/nathan/data/ccrcc/TCGA_KIRC/'
slide_path += 'TCGA-A3-3346-01Z-00-DX1.95280216-fd71-4a03-b452-6e3d667f2542.svs'
preprocess_fn = lambda x: (x * (2/255.)) - 1
s = Slide(slide_path    = slide_path,
          process_mag   = 5,
          process_size  = 128,
          preprocess_fn = preprocess_fn
          )
s.print_info()

ds = tf.data.Dataset.from_generator(generator=s.generator, output_types=tf.float32)
ds = tfe.Iterator(ds)

for idx, x in enumerate(ds):
    print x.shape, x.dtype
    # cv2.imwrite('debug/{}.jpg'.format(idx), x.numpy()[:,:,::-1])
