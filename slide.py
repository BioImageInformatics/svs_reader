"""
Args:
  slide_path: string
  process_mag: string
  process_size: int

Returns:
  Slide object

https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator
"""
from __future__ import print_function
from foreground import get_foreground
from normalize import reinhard
from openslide import OpenSlide
import numpy as np
import cv2

class Slide(object):

    # set up constants parse slide information
    # set up output image
    def __init__(self, **kwargs):
        slide_defaults = {
            'slide_path': None,
            'low_level_mag': 5,
            'preprocess_fn': lambda x: x,  ## ID
            'process_mag': 10,
            'process_size': 256,
            'normalize_fn': reinhard,
            'oversample_factor': 1.25,
            'output_types': [],
            'output_res': '5x',
            'verbose': False}
        slide_defaults.update(kwargs)
        for key, val in slide_defaults.items():
            setattr(self, key, val)

        self.svs = self._parse_svs_info()
        self.foreground = get_foreground(self.svs)
        self._get_load_params()
        self.tile()

        ## Reconstruction params
        self._get_place_params()
        self.output_imgs = {}


    def _get_low_level_index(self):
        ## get the info to read/write the low-level image more efficiently
        ## This operation is instead of simply using the lowest resolution
        ## size to write output.
        ## Use the scanned power, and requested magnification to find the downsample factor
        pass

    # Returns the OpenSlide object
    # Populates a dict with:
    # - scanning power
    # - downsample fractions
    # - low-level dimensions
    def _parse_svs_info(self):
        svs = OpenSlide(self.slide_path)
        scan_power = int(svs.properties['aperio.AppMag'])
        level_count = svs.level_count
        high_power_dim = svs.level_dimensions[0][::-1]
        low_power_dim = svs.level_dimensions[-1][::-1]

        #if scan_power == 20 and level_count ==4:
        #    raise Exception('Malformed slide. {}'.format(self.slide_path))

        if self.verbose:
            print('Slide: %s' % self.slide_path)
            print('\t power: %d' % scan_power)
            print('\t levels: %d' % level_count)
            print('\t high_power_dim: %d %d' % (high_power_dim))
            print('\t low_power_dim: %d %d' % (low_power_dim))

        self.slide_info = {
            'scan_power': scan_power,
            'level_count': level_count,
            'high_power_dim': high_power_dim,
            'low_power_dim': low_power_dim,
            'level_dimensions': svs.level_dimensions }
        return svs


    def close(self):
        print('Closing slide')
        self.foreground = []
        self.output_imgs = []
        self.svs.close()


    # Set up the output image to the same size as the level-0 shape
    def initialize_output(self, name, dim, mode='full'):
        ## Initialize an image for dimensions preserving output
        if mode=='full':
            h,w = self.foreground.shape[:2]
            output_img = np.zeros((int(h), int(w), dim), dtype=np.float32)
            self.output_imgs[name] = output_img

        ## Initialize an image for one-value-per-tile output (dimensions reducing)
        elif mode=='tile':
            y = len(self.y_coord)
            x = len(self.x_coord)
            output_img = np.zeros((y, x, dim), dtype=np.float32)
            self.output_imgs[name] = output_img

        self.output_types.append(name)


    def _get_load_size(self, process_size, loading_level, downsample):
        ds_load_level = int(self.svs.level_downsamples[loading_level])

        if self.verbose:
            print('Requested processing size: {}'.format(process_size))
            print('Estimated loading from level: {}'.format(loading_level))
            print('Downsample at estimated level: {}'.format(ds_load_level))

        self.ds_load_level = ds_load_level

        ## scan @ Nx ; request 10x
        ## scan @ Nx ; request 5x
        if ds_load_level == downsample:
            if self.verbose:
                print('Loading size: {} ({}x processing size)'.format(
                    process_size, 1))
            return process_size, 1

        ## scan @ 40x; request 20x
        if ds_load_level < downsample:
            ratio = int(downsample / ds_load_level)
            if self.verbose:
                print('Loading size: {} ({}x processing size)'.format(
                    process_size*ratio, ratio))
            return process_size*ratio, 1./ratio


    # Logic translating slide params and requested process_mag into read_region args
    def _get_load_params(self):
        ## Add a small number to the requested downsample because often we're off by some.
        EPS = 1e-3
        downsample = int(self.slide_info['scan_power'] / self.process_mag)
        loading_level = self.svs.get_best_level_for_downsample(downsample+EPS)
        load_level_dims = self.svs.level_dimensions[loading_level][::-1]
        loading_size, post_load_resize = self._get_load_size(self.process_size,
            loading_level, downsample)

        if self.verbose:
            print('Slide scanned at {} magnification'.format(self.slide_info['scan_power']))
            print('Requested processing at {} magnification'.format(self.process_mag))
            print('Downsample ~ {}'.format(downsample))
            print('Load from level {}'.format(loading_level))
            print('Level {} dimensions: {}'.format(loading_level, load_level_dims))

        self.downsample = downsample
        self.loading_level = loading_level
        self.load_level_dims = load_level_dims
        self.loading_size = loading_size
        self.post_load_resize = post_load_resize


    # Logic translating processing size into reconstruct() args
    def _get_place_params(self):
        ## Place w.r.t. level 0
        ## We have process downsample.. and downsample w.r.t. Last level
        ds_low_level = int(self.svs.level_downsamples[-1])
        place_downsample = self.downsample / float(ds_low_level)
        self.ds_low_level = ds_low_level
        place_size = int(self.process_size * place_downsample)
        if self.verbose:
            print('Placing size: {}'.format(place_size))

        self.place_size = place_size

        place_list = []
        for coords in self.tile_list:
            y, x = coords
            place_list.append([
                int(y*(1./ds_low_level)),
                int(x*(1./ds_low_level)) ])
        self.place_list = place_list


    ## Returns the parameters needed to replicate the corresponding
    ## call to _read_tile
    ## return y1, y2, x1, x2, level, downsample
    def _read_region_args(self, coords):
        y1, x1 = coords
        # y1 = int(y1 * self.post_load_resize)
        # x1 = int(x1 * self.post_load_resize)
        y1 = int(y1 / self.ds_load_level)
        x1 = int(x1 / self.ds_load_level)
        y2 = int(y1 + self.loading_size * self.post_load_resize)
        x2 = int(x1 + self.loading_size * self.post_load_resize)
        level = self.loading_level
        downsample = self.post_load_resize
        return y1, y2, x1, x2, level, downsample

    # Call openslide.read_region on the slide
    # with all the right settings: level, dimensions, etc.
    def _read_tile(self, coords):
        y, x = coords
        size = (self.loading_size, self.loading_size)
        img = self.svs.read_region((x, y), self.loading_level, size)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = self.normalize_fn(img)
        img = cv2.resize(img, dsize=(0,0), fx=self.post_load_resize,
            fy=self.post_load_resize)
        img = self.preprocess_fn(img)
        return img


    def generate_index(self):
        for idx, _ in enumerate(self.tile_list):
            yield idx


    ## TODO add live skipping of white area
    def generator(self):
        for idx, coords in enumerate(self.tile_list):
            img = self._read_tile(coords)
            yield img, idx


    # Generate a list of foreground tiles
    # 1. Call get_foreground
    # 2. Estimate tiles, with/without overlap according to settings
    # 3. Reject background-only tiles
    def _find_all_tiles(self):
        load_y, load_x = self.load_level_dims
        est_y = int(load_y / self.loading_size)
        est_x = int(load_x / self.loading_size)

        y_coord = np.linspace(0, load_y-self.loading_size,
            int(est_y*self.oversample_factor), dtype=np.int64)
        x_coord = np.linspace(0, load_x-self.loading_size,
            int(est_x*self.oversample_factor), dtype=np.int64)

        if self.verbose:
            print('Estimated w={} x h={} tiles'.format(est_x, est_y))
            print('With oversample ~ {}, split x={} x y={}'.format(
                self.oversample_factor, len(x_coord), len(y_coord) ))

        self.y_coord = y_coord
        self.x_coord = x_coord


    def _reject_background(self):
        yc, xc = self.y_coord, self.x_coord
        foreground_ds = cv2.resize(self.foreground,
                                   dsize=( len(xc), len(yc) ),
                                   interpolation=cv2.INTER_NEAREST)

        tile_idx = 0
        tile_list = []
        self.ds_tile_map = np.zeros((len(yc), len(xc)), dtype=np.uint64)-1
        for yi, yy in enumerate(yc):
            for xi, xx in enumerate(xc):
                if foreground_ds[yi, xi]==1:
                    self.ds_tile_map[yi, xi] = tile_idx
                    tile_idx += 1
                    tile_list.append(
                        [yy*self.ds_load_level,
                         xx*self.ds_load_level])

        if self.verbose:
            print('Started with {} candidate tiles'.format(len(yc)*len(xc)))
            print('Got {} foreground tiles'.format(len(tile_list)))

        self.tile_list = tile_list

    def tile(self):
        self.tile_list = self._find_all_tiles()
        self._reject_background()


    # place x into location, doing whatever downsampling is needed
    def place(self, x, idx, name, mode='full'):
        if mode=='full':
            place_coord = self.place_list[idx]
            y0, x0 = place_coord
            x1 = x0 + int(self.place_size)
            y1 = y0 + int(self.place_size)
            x = cv2.resize(x, dsize=(int(self.place_size), int(self.place_size)))
            self.output_imgs[name][y0:y1, x0:x1, :] += x
        elif mode=='tile':
            location = self.ds_tile_map == idx
            self.output_imgs[name][location] = x

    def place_batch(self, xs, idxs, name, mode='full'):
        for x , idx in zip(xs,idxs):
            self.place(x, idx, name, mode=mode)


    ## Valid probability distribution sums to 1.
    ## We can tell where the overlaps are by finding areas that sum > 1
    def get_overlapping_images(self):
        prob_img = self.output_imgs['prob']
        prob_sum = np.sum(prob_img, axis=-1)

        self.twice_overlapping = prob_sum == 2
        self.thrice_overlapping = prob_sum == 3
        self.quad_overlapping = prob_sum == 4
        print('Found {} areas with 2x coverage'.format(self.twice_overlapping.sum()))
        print('Found {} areas with 3x coverage'.format(self.thrice_overlapping.sum()))
        print('Found {} areas with 4x coverage'.format(self.quad_overlapping.sum()))


    # colorize, and write out
    def make_outputs(self):
        self.get_overlapping_images()
        for key, img in self.output_imgs.items():
            img[self.twice_overlapping]  = img[self.twice_overlapping] / 2.
            img[self.thrice_overlapping] = img[self.thrice_overlapping] / 3.
            img[self.quad_overlapping]   = img[self.quad_overlapping] / 4.
            self.output_imgs[key] = img


    def print_info(self):
        print('\n======================= SLIDE ======================')
        print('|')
        for key, val in sorted(self.__dict__.items()):
            if 'list' in key:
                print('|\t {}:\n|\t\t\tlength: {}'.format(key, len(val)))
                continue

            if type(val) is np.ndarray:
                print('|\t {}:\n|\t\t\tshape: {}'.format(key, val.shape))
                continue

            if key == 'output_imgs':
                try:
                    for vk, vv in val.items():
                        print('|\t {}:\n|\t\t\t{}: {}'.format(key, vk, vv.shape))
                except:
                    print('|\t {}:\n|\t\t\tlen: {}'.format(key, len(val)))
                continue

            print('|\t {}:\n|\t\t\t{}'.format(key, val))
        print('|')
        print('======================= SLIDE ======================\n')
