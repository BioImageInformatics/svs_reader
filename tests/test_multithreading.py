from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import cv2
import time

from ..slide import Slide

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

"""
https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator
https://www.tensorflow.org/programmers_guide/datasets
https://www.tensorflow.org/api_docs/python/tf/data/Dataset
"""

# slide_path = '/home/nathan/data/ccrcc/TCGA_KIRC/'
# slide_path += 'TCGA-A3-3346-01Z-00-DX1.95280216-fd71-4a03-b452-6e3d667f2542.svs'

# slide_path = '/home/nathan/data/gleason_grade/wsi/s10-3220-001.svs'
# slide_path = '/home/nathan/biobank/_DURHAM_SLIDES_/_Cases_/S10-3220/s10-3220-001.svs'
slide_path = '/dev/shm/s13_2243-016.svs'

preprocess_fn = lambda x: ((x * (2/255.)) - 1).astype(np.float32)
s = Slide(slide_path    = slide_path,
          process_mag   = 10,
          process_size  = 256,
          preprocess_fn = preprocess_fn
          )
s.print_info()

# for ix in s.generate_index():
#     print ix

def wrapped_fn(idx):
    coords = s.tile_list[idx]
    img = s._read_tile(coords)
    return img, idx

def read_region_at_index(idx):
    return tf.py_func(func     = wrapped_fn,
                      inp      = [idx],
                      Tout     = [tf.float32, tf.int64],
                      stateful = False)

# for ix in s.generate_index():
#     img, idx = wrapped_fn(ix)
#     print img.shape, idx

with tf.Session(config=config) as sess:
    ds = tf.data.Dataset.from_generator(generator=s.generate_index,
        output_types=tf.int64)
    ds = ds.map(read_region_at_index, num_parallel_calls=8)
    ds = ds.prefetch(2048)
    ds = ds.batch(16)
    iterator = ds.make_one_shot_iterator()
    img, idx = iterator.get_next()

    print('Starting')
    tstart = time.time()
    while True:
        try:
            img_, idx_ = sess.run([img, idx])
            print(idx_, img_.shape, img_.dtype)
        except Exception as e:
            print(e)
            break

    print('Finished in {}s'.format(time.time() - tstart))
