import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import sys
import cv2

sys.path.insert(0, '..')
from slide import slide

config = tf.configproto()
config.gpu_options.allow_growth = true
tfe.enable_eager_execution(config=config)

print '\nslide at 5x'
s = slide(slide_path='/home/nathan/data/ccrcc/tcga_kirc/tcga-a3-3346-01z-00-dx1.95280216-fd71-4a03-b452-6e3d667f2542.svs',
        process_mag=5, )
s.print_info()

ds = tf.data.Dataset.from_generator(generator=s.generator, output_types=tf.float32)
ds = tfe.iterator(ds)

for idx, x in enumerate(ds):
    print x.shape
    cv2.imwrite('debug/{}.jpg'.format(idx), x.numpy()[:,:,::-1])
