"""
Generate diagnostic plots for a single radio-transient candidate.

This script loads a filterbank segment around a candidate, produces four
visualisations, and saves them to disk:

1. Channels vs. Time samples (raw chunk)
2. DM index vs. Time samples (DM–time map)
3. Dedispersed Channels vs. Time samples
4. Flux (arbitrary units) vs. Time samples (time series collapsed over channels)

It relies on a `Candidate` object (from `your.candidate`) that exposes:
- `get_chunk()` to extract a time–frequency window into `cand.data`
- `dmtime()` to compute a DM–time map into `cand.dmt`
- `dedisperse()` to compute dedispersed data into `cand.dedispersed`

Command-line arguments
----------------------
-o1, --output_plot1 : str, optional
    Output filename for the Channels vs. Time image (default: ``channels_vs_time.png``).
-o2, --output_plot2 : str, optional
    Output filename for the DM index vs. Time image (default: ``dm_index_vs_time.png``).
-o3, --output_plot3 : str, optional
    Output filename for the Dedispersed Channels vs. Time image (default: ``dispersed_channels_vs_time.png``).
-o4, --output_plot4 : str, optional
    Output filename for the Flux vs. Time line plot (default: ``flux_vs_time.png``).

Notes
-----
- HDF5 file locking is disabled by setting the environment variable
  ``HDF5_USE_FILE_LOCKING=FALSE`` to avoid locking issues on shared filesystems.
- Input file path and candidate parameters are currently hard-coded in this script
  (see ``fil_file``, ``dm``, ``tcand``, ``width``, ``snr``); adjust as needed.
- Figures are displayed via Matplotlib and also saved to the specified output paths.

Example
-------
Save all four plots to default filenames:

    python plot_candidate.py

Specify custom filenames:

    python plot_candidate.py \\
        -o1 chan_time.png -o2 dm_time.png -o3 dedisp.png -o4 flux.png
"""

import matplotlib
from your.candidate import Candidate
import numpy as np
from scipy.signal import detrend
import os
import argparse
from your.utils.plotter import plot_h5

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import pylab as plt
import logging

logger = logging.getLogger()
logger = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(threadName)s - %(levelname)s -" " %(message)s",
)

def validate_file(file):
    """
    Validate that a path points to an existing file (for argparse).

    Parameters
    ----------
    file : str
        Filesystem path to validate.

    Returns
    -------
    str
        The same `file` path if it exists.

    Raises
    ------
    argparse.ArgumentTypeError
        If the file does not exist.
    """
    if not os.path.exists(file):
    # Argparse uses the ArgumentTypeError to give a rejection message like:
    # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("The file does not exist.")
    return file

parser = argparse.ArgumentParser(description="Read file from Command line.")
parser.add_argument("-i", "--input", dest="input_filename", required=True, type=validate_file, help="input filterbank file", metavar="FILE")
parser.add_argument("-dm", "--dm", dest="dm_value", required=True, help="dm value")
parser.add_argument("-toa","--ToA", dest="time_of_arrival", required=True, help="time of arrival")
parser.add_argument("-w", "--width", dest="width", required=False, default=0.0001, help="width")
parser.add_argument("-snr", "--snr", dest="snr", required=True, help="signal to noise ratio")
parser.add_argument("-o1", "--output_plot1", dest="output_1", required=False, default="channels_vs_time.png", help="output plot 1 filename (default: channels_vs_time.png)", metavar="FILE")
parser.add_argument("-o2", "--output_plot2", dest="output_2", required=False, default="dm_index_vs_time.png", help="output plot 2 filename (default: dm_index_vs_time.png)", metavar="FILE")
parser.add_argument("-o3", "--output_plot3", dest="output_3", required=False, default="dispersed_channels_vs_time.png", help="output plot 3 filename (default: dispersed_channels_vs_time.png)", metavar="FILE")
parser.add_argument("-o4", "--output_plot4", dest="output_4", required=False, default="flux_vs_time.png", help="output plot 4 filename (default: flux_vs_time.png)", metavar="FILE")

args = parser.parse_args()

fil_file = args.input_filename

cand = Candidate(
    fp=fil_file,
    dm=args.dm_value,
    tcand=args.time_of_arrival,
    width=args.width,
    label=0,
    snr=args.snr,
    min_samp=512,
    device=0,
)

cand.get_chunk()
print(cand.data, cand.data.shape, cand.dtype)

plt.imshow(cand.data.T, aspect="auto", interpolation=None)
plt.ylabel("Channels")
plt.xlabel("Time Samples")
output_file1 = args.output_1
plt.savefig(output_file1)
plt.show()


cand.dmtime()

plt.imshow(cand.dmt, aspect="auto", interpolation=None)
plt.ylabel("DM index")
plt.xlabel("Time Samples")
output_file2 = args.output_2
plt.savefig(output_file2)
plt.show()


cand.dedisperse()

plt.imshow(cand.dedispersed.T, aspect="auto", interpolation=None)
plt.ylabel("Channels")
plt.xlabel("Time Samples")
output_file3 = args.output_3
plt.savefig(output_file3)
plt.show()

plt.plot(cand.dedispersed.T.sum(0))
plt.xlabel("Time Samples")
plt.ylabel("Flux (Arb. Units)")
output_file4 = args.output_4
plt.savefig(output_file4)
plt.show()
