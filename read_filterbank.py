"""
Load a SIGPROC `.fil` file with `your`, preview header info, and plot a dynamic spectrum.

This script:
1) opens a filterbank file via `your.Your`,
2) prints the parsed header,
3) reads a chunk of data (`nsamp=4096` starting at `nstart=0`), and
4) renders a time–frequency image (channels × time) saved as `frb_plot.png`.

Inputs
------
- The relative path to a `.fil` file is hard-coded as ``../filterbank_file.fil``.

Outputs
-------
- `frb_plot.png` — an image of the dedispersed (raw) dynamic spectrum chunk.

Dependencies
------------
- `your` (Python package for radio astronomy filterbank/PSRFITS I/O)
- `matplotlib`

Notes
-----
- `your_object.get_data` returns an array with shape `(nchans, nsamp)`. The data is
  transposed for display so that time is on the x-axis and frequency channels on the y-axis.
"""

import your
import os
import tempfile
import pylab as plt
from urllib.request import urlretrieve

fil_file='../filterbank_file.fil'

your_object = your.Your(fil_file)

print(your_object.your_header)

data = your_object.get_data(nstart=0, nsamp=4096)
data.shape

plt.figure(figsize=(8, 6))
plt.imshow(data.T, aspect="auto")
plt.xlabel("Time Samples")
plt.ylabel("Frequency Channels")
plt.colorbar()
plt.savefig("frb_plot.png")
plt.show()
