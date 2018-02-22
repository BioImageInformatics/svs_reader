"""
Args:
  slide_path: string
  process_mag: string
  process_size: int

Returns:
  Slide object

"""
from foreground import get_foreground
from normalize import reinhard
from openslide import OpenSlide
import numpy as np
import cv2

class Slide(object):
    slide_defaults = {
        'slide_path': None,
        'process_mag': 10,
        'process_size': 256,
        'oversample_factor': 1.1,
    }

    # set up constants parse slide information
    # set up output image
    def __init__(self, **kwargs):
        self.slide_defaults.update(kwargs)
        for key, val in self.slide_defaults.items():
            setattr(self, key, val)

        self.svs = self._parse_svs_info()
        self.foreground = get_foreground(self.svs)
        self._get_load_params()
        self.tile()


    # Returns the OpenSlide object
    # Populates a dict with:
    # - scanning power
    # - downsample fractions
    # - low-level dimensions
    def _parse_svs_info(self):
        svs = OpenSlide(self.slide_path)

        scan_power = int(svs.properties['aperio.AppMag'])
        level_count = svs.level_count
        high_power_dim = svs.level_dimensions[0]
        low_power_dim = svs.level_dimensions[-1]

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

    # Set up the output image to the same size as the level-0 shape
    def _initialize_output(self):
        pass


    def _get_load_size(self, process_size, loading_level, downsample):
        print 'Requested processing size: {}'.format(process_size)
        ds_load_level = int(self.svs.level_downsamples[loading_level])
        print 'Estimated loading from level: {}'.format(loading_level)
        print 'Downsample at estimated level: {}'.format(ds_load_level)

        self.ds_load_level = ds_load_level

        ## scan @ Nx ; request 10x
        ## scan @ Nx ; request 5x
        if ds_load_level == downsample:
            print 'Loading size: {} ({}x processing size)'.format(
                process_size, 1)
            return process_size, 1

        ## scan @ 40x; request 20x
        if ds_load_level < downsample:
            ratio = int(downsample / ds_load_level)
            print 'Loading size: {} ({}x processing size)'.format(
                process_size*ratio, ratio)
            return process_size*ratio, 1./ratio



    # Logic translating slide params and requested process_mag into read_region args
    def _get_load_params(self):
        print 'Slide scanned at {} magnification'.format(self.slide_info['scan_power'])
        print 'Requested processing at {} magnification'.format(self.process_mag)

        downsample = int(self.slide_info['scan_power'] / self.process_mag)
        print 'Downsample ~ {}'.format(downsample)

        loading_level = self.svs.get_best_level_for_downsample(downsample)
        print 'Load from level {}'.format(loading_level)

        load_level_dims = self.svs.level_dimensions[loading_level]
        print 'Level {} dimensions: {}'.format(loading_level, load_level_dims)

        loading_size, post_load_resize = self._get_load_size(self.process_size,
            loading_level, downsample)

        self.downsample = downsample
        self.loading_level = loading_level
        self.load_level_dims = load_level_dims
        self.loading_size = loading_size
        self.post_load_resize = post_load_resize


    # Logic translating processing size into reconstruct() args
    def _get_place_params():
        pass

    # Call openslide.read_region on the slide
    # with all the right settings: level, dimensions, etc.
    def _read_tile(self, coords):
        x, y = coords
        size = (self.loading_size, self.loading_size)
        img = self.svs.read_region((x, y), self.loading_level, size)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(img, dsize=(0,0), fx=self.post_load_resize,
            fy=self.post_load_resize)
        return img

    # Like this:
    def generator(self):
        for coords in self.tile_list:
            yield self._read_tile(coords)


    # Generate a list of foreground tiles
    # 1. Call get_foreground
    # 2. Estimate tiles, with/without overlap according to settings
    # 3. Reject background-only tiles
    def _find_all_tiles(self):
        load_w, load_h = self.load_level_dims
        est_w = int(load_w / self.loading_size)
        est_h = int(load_h / self.loading_size)

        w_coord = np.linspace(0, load_w-self.loading_size,
            est_w*self.oversample_factor, dtype=np.int64)
        h_coord = np.linspace(0, load_h-self.loading_size,
            est_h*self.oversample_factor, dtype=np.int64)

        print 'Estimated w={} x h={} tiles'.format(est_w, est_h)
        print 'With oversample ~ {}, split w={} x h={}'.format(
            self.oversample_factor, len(w_coord), len(h_coord) )

        self.w_coord = w_coord
        self.h_coord = h_coord


    def _reject_background(self):
        wc, hc = self.w_coord, self.h_coord
        foreground_ds = cv2.resize(self.foreground,
            dsize=( len(hc), len(wc) ),
            interpolation=cv2.INTER_NEAREST)

        tile_idx = 0
        tile_list = []
        for wi, ww in enumerate(wc):
            for hi, hh in enumerate(hc):
                if foreground_ds[wi, hi]==1:
                    tile_list.append(
                        [ww*self.ds_load_level,
                         hh*self.ds_load_level])

        print 'Started with {} candidate tiles'.format(len(wc)*len(hc))
        print 'Got {} foreground tiles'.format(len(tile_list))
        self.tile_list = tile_list


    def tile(self):
        self.tile_list = self._find_all_tiles()
        self._reject_background()

    # place x into location, doing whatever downsampling is needed
    def reconstruct(x, location):
        pass


    # colorize, and write out
    def make_outputs():
        pass


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

            print '|\t {}:\n|\t\t\t{}'.format(key, val)
        print '|'
        print '======================= SLIDE ======================\n'
