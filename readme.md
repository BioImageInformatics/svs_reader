# SVS-READER

An object for interfacing __usually__ Aperio's SVS slide images. Wraps OpenSlide.

Find foreground, tile foreground, yield tiles. See tests for some example scripts.

### Installation
Prerequisites:
```
openslide-python
opencv
numpy
```

To install:
```
/usr/env/bin/pip install -e .
```

To use:
```
$ python
>>> from svs_reader import Slide
```

### License
Please provide citation if you use this library for your research.

Copyright 2017 BioImageInformatics Lab, Cedars-Sinai Medical Center

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
