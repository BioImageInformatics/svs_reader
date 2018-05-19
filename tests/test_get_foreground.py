from __future__ import print_function
from openslide import OpenSlide
from ..foreground import get_foreground
from matplotlib import pyplot as plt

svs = OpenSlide('/home/nathan/data/ccrcc/TCGA_KIRC/TCGA-A3-3380-01Z-00-DX1.67e52764-4c94-42b5-a725-4f61a86b49fe.svs')
fg = get_foreground(svs)

print(fg.shape)
plt.matshow(fg)
plt.colorbar()
plt.show()
