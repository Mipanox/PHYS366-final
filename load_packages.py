"""
Adding path to downloaded codes:
- The old 'agpy' tools developed by keflavich:
   https://github.com/keflavich/image_tools
- The 'lmc' code developed by Adam Mantz
   https://github.com/abmantz/lmc
"""
from __future__ import division
import sys
## search the folder, add to the beginning to prevent name conflicting
sys.path.insert(0, './codes/')

##
import lmc
from psds import PSD2
from astropy.utils.data import get_readable_fileobj
from astropy.io import fits
import numpy as np
from scipy.fftpack import ifft
from scipy import signal
import matplotlib.pyplot as plt
