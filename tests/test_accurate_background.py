from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import cv2
import time

sys.path.insert(0, '..')
from slide import Slide

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

"""
https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator
https://www.tensorflow.org/programmers_guide/datasets
https://www.tensorflow.org/api_docs/python/tf/data/Dataset
"""

# slide_path = '/media/ing/D/svs/TCGA_KIRC/TCGA-A3-3306-01Z-00-DX1.bfd320d3-f3ec-4015-b34a-98e9967ea80d.svs'
slide_path = '/mnt/slowdata/slide_data/VA_PNBX/SP 02-4466 L3.svs'

print('Testing "accurate" background method')
tstart = time.time()
preprocess_fn = lambda x: (x * 1/255.).astype(np.float32)
svs = Slide(slide_path    = slide_path,
          process_mag     = 5,
          process_size    = 96,
          oversample_factor = 1.75,
          preprocess_fn   = preprocess_fn,
          background_speed = 'accurate',
          background_threshold = 210,
          background_pct = 0.15,
          verbose=True
          )
svs.print_info()
svs.initialize_output('features', dim=3, mode='tile')
print('Initialized slide object in {}s'.format(time.time() - tstart))

def wrapped_fn(idx):
    coords = svs.tile_list[idx]
    img = svs._read_tile(coords)
    return img, idx

def read_region_at_index(idx):
    return tf.py_func(func     = wrapped_fn,
                      inp      = [idx],
                      Tout     = [tf.float32, tf.int64],
                      stateful = False)

def feature_fn(img):
    r, g, b = np.split(img, 3, axis=2)
    output = np.asarray([np.mean(r), np.mean(g), np.mean(b)])
    return np.expand_dims(output, axis=0)


with tf.Session(config=config) as sess:
    ds = tf.data.Dataset.from_generator(generator=svs.generate_index,
        output_types=tf.int64)
    ds = ds.map(read_region_at_index, num_parallel_calls=4)
    ds = ds.prefetch(512)
    iterator = ds.make_one_shot_iterator()
    img, idx = iterator.get_next()

    print('Starting')
    tstart = time.time()
    while True:
        try:
            img_, idx_ = sess.run([img, idx])
            features = feature_fn(img_)
            svs.place(features, idx_, name='features', mode='tile')
        except tf.errors.OutOfRangeError:
            print('End')
            break

    print('Finished in {}s'.format(time.time() - tstart))

    img_out = svs.output_imgs['features']
    cv2.imwrite('test_accurate_background.jpg', img_out * (255. / img_out.max()))
