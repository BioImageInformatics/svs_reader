"""
Args:
  slide_path: string
  process_mag: string
  process_size: int

Returns:
  Slide object

https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator
"""
from foreground import get_foreground
from normalize import reinhard
from openslide import OpenSlide
import numpy as np
import cv2

class Slide(object):
    slide_defaults = {
        'slide_path': None,
        'low_level_mag': 5,
        'preprocess_fn': lambda x: x,  ## ID
        'process_mag': 10,
        'process_size': 256,
        'normalize_fn': reinhard,
        'oversample_factor': 1.25,
        'output_types': ['prob'],
        'verbose': False}

    # set up constants parse slide information
    # set up output image
    def __init__(self, **kwargs):
        self.slide_defaults.update(kwargs)
        for key, val in self.slide_defaults.items():
            setattr(self, key, val)

        self.svs = self._parse_svs_info()
        # self.low_level_index = self.get_low_level_index()
        self.foreground = get_foreground(self.svs, low_level_index=2)
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

        if self.verbose:
            print 'Slide: %s' % self.slide_path
            print '\t power: %d' % scan_power
            print '\t levels: %d' % level_count
            print '\t high_power_dim: %d %d' % (high_power_dim)
            print '\t low_power_dim: %d %d' % (low_power_dim)

        self.slide_info = {
            'scan_power': scan_power,
            'level_count': level_count,
            'high_power_dim': high_power_dim,
            'low_power_dim': low_power_dim }
        return svs


    def close(self):
        print 'Closing slide'
        self.foreground = []
        self.output_imgs = []
        self.svs.close()


    # Set up the output image to the same size as the level-0 shape
    def initialize_output(self, name, dim):
        h,w = self.foreground.shape[:2]
        # h *= self.low_level_upsample
        # w *= self.low_level_upsample
        output_img = np.zeros((int(h), int(w), dim), dtype=np.float32)

        if self.verbose:
            print 'Initialized output image shape: {}'.format(output_img.shape)

        self.output_imgs[name] = output_img


    def _get_load_size(self, process_size, loading_level, downsample):
        ds_load_level = int(self.svs.level_downsamples[loading_level])

        if self.verbose:
            print 'Requested processing size: {}'.format(process_size)
            print 'Estimated loading from level: {}'.format(loading_level)
            print 'Downsample at estimated level: {}'.format(ds_load_level)

        self.ds_load_level = ds_load_level

        ## scan @ Nx ; request 10x
        ## scan @ Nx ; request 5x
        if ds_load_level == downsample:
            if self.verbose:
                print 'Loading size: {} ({}x processing size)'.format(
                    process_size, 1)
            return process_size, 1

        ## scan @ 40x; request 20x
        if ds_load_level < downsample:
            ratio = int(downsample / ds_load_level)
            if self.verbose:
                print 'Loading size: {} ({}x processing size)'.format(
                    process_size*ratio, ratio)
            return process_size*ratio, 1./ratio


    # Logic translating slide params and requested process_mag into read_region args
    def _get_load_params(self):

        downsample = int(self.slide_info['scan_power'] / self.process_mag)
        loading_level = self.svs.get_best_level_for_downsample(downsample)
        load_level_dims = self.svs.level_dimensions[loading_level][::-1]
        loading_size, post_load_resize = self._get_load_size(self.process_size,
            loading_level, downsample)

        if self.verbose:
            print 'Slide scanned at {} magnification'.format(self.slide_info['scan_power'])
            print 'Requested processing at {} magnification'.format(self.process_mag)
            print 'Downsample ~ {}'.format(downsample)
            print 'Load from level {}'.format(loading_level)
            print 'Level {} dimensions: {}'.format(loading_level, load_level_dims)

        self.downsample = downsample
        self.loading_level = loading_level
        self.load_level_dims = load_level_dims
        self.loading_size = loading_size
        self.post_load_resize = post_load_resize


    # Logic translating processing size into reconstruct() args
    def _get_place_params(self):
        ## Place w.r.t. level 0
        ## We have process downsample.. and downsample w.r.t. Last level
        ds_low_level = int(self.svs.level_downsamples[2])
        place_downsample = self.downsample / float(ds_low_level)
        self.ds_low_level = ds_low_level
        place_size = int(self.process_size * place_downsample)
        if self.verbose:
            print 'Placing size: {}'.format(place_size)

        self.place_size = place_size

        place_list = []
        for coords in self.tile_list:
            y, x = coords
            place_list.append([
                int(y*(1./ds_low_level)),
                int(x*(1./ds_low_level)) ])
        self.place_list = place_list


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


    def generator(self):
        for coords in self.tile_list:
            yield self._read_tile(coords)


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
            print 'Estimated w={} x h={} tiles'.format(est_w, est_h)
            print 'With oversample ~ {}, split w={} x h={}'.format(
                self.oversample_factor, len(w_coord), len(h_coord) )

        self.y_coord = y_coord
        self.x_coord = x_coord


    def _reject_background(self):
        yc, xc = self.y_coord, self.x_coord
        foreground_ds = cv2.resize(self.foreground,
            dsize=( len(xc), len(yc) ),
            interpolation=cv2.INTER_NEAREST)

        tile_idx = 0
        tile_list = []
        for yi, yy in enumerate(yc):
            for xi, xx in enumerate(xc):
                if foreground_ds[yi, xi]==1:
                    tile_list.append(
                        [yy*self.ds_load_level,
                         xx*self.ds_load_level])

        if self.verbose:
            print 'Started with {} candidate tiles'.format(len(yc)*len(xc))
            print 'Got {} foreground tiles'.format(len(tile_list))

        self.tile_list = tile_list

    def tile(self):
        self.tile_list = self._find_all_tiles()
        self._reject_background()

    # place x into location, doing whatever downsampling is needed
    def place(self, x, idx, name):
        place_coord = self.place_list[idx]
        y0, x0 = place_coord
        x1 = x0 + int(self.place_size)
        y1 = y0 + int(self.place_size)
        # print 'Resize {} --> {}'.format(x.shape, self.place_size),
        x = cv2.resize(x, dsize=(int(self.place_size),
            int(self.place_size)))
        # print 'placing {}:{}, {}:{} ; {}'.format(x0, x1, y0, y1, x.shape)
        self.output_imgs[name][y0:y1, x0:x1, :] += x

    def place_batch(self, xs, idxs, name):
        for x , idx in zip(xs,idxs):
            self.place(x, idx, name)

    ## Valid probability distribution sums to 1.
    ## We can tell where the overlaps are by finding areas that sum > 1
    def get_overlapping_images(self):
        prob_img = self.output_imgs['prob']
        prob_sum = np.sum(prob_img, axis=-1)

        self.twice_overlapping = prob_sum == 2
        self.thrice_overlapping = prob_sum == 3
        self.quad_overlapping = prob_sum == 4
        print 'Found {} areas with 2x coverage'.format(self.twice_overlapping.sum())
        print 'Found {} areas with 3x coverage'.format(self.thrice_overlapping.sum())
        print 'Found {} areas with 4x coverage'.format(self.quad_overlapping.sum())


    # colorize, and write out
    def make_outputs(self):
        self.get_overlapping_images()
        for key, img in self.output_imgs.items():
            img[self.twice_overlapping]  = img[self.twice_overlapping] / 2.
            img[self.thrice_overlapping] = img[self.thrice_overlapping] / 3.
            img[self.quad_overlapping]   = img[self.quad_overlapping] / 4.
            self.output_imgs[key] = img


    def print_info(self):
        print '\n======================= SLIDE ======================'
        print '|'
        for key, val in sorted(self.__dict__.items()):
            if 'list' in key:
                print '|\t {}:\n|\t\t\tlength: {}'.format(key, len(val))
                continue

            if type(val) is np.ndarray:
                print '|\t {}:\n|\t\t\tshape: {}'.format(key, val.shape)
                continue

            if key == 'output_imgs':
                # print '|\t {}:\n|\t\t\tlen: {}'.format(key, len(val))
                continue

            print '|\t {}:\n|\t\t\t{}'.format(key, val)
        print '|'
        print '======================= SLIDE ======================\n'
