import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import sys
import cv2
import time

sys.path.insert(0, '..')
from slide import Slide

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print '\nslide at 5x'
s = Slide(slide_path='/home/nathan/data/ccrcc/TCGA_KIRC/TCGA-A3-3346-01Z-00-DX1.95280216-fd71-4a03-b452-6e3d667f2542.svs',
        process_mag=5, )
s.print_info()

with tf.Session(config=config) as sess:
    ds = tf.data.Dataset.from_generator(generator=s.generator, output_types=tf.float32)
    # ds = ds.map(lambda x: x, num_parallel_calls=8)
    # ds = ds.prefetch(128)
    iterator = ds.make_one_shot_iterator()
    img = iterator.get_next()

    tstart = time.time()
    for x in range(len(s.tile_list)):
        img_ = sess.run(img)

    print 'Finished in {}s'.format(time.time() - tstart)
