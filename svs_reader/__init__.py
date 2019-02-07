name = 'svs_reader'

from .slide import Slide
from .normalize import reinhard

__all__ = [
    'Slide',
    'reinhard'
]
