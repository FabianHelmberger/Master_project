"""
    Real scalar functions and types
"""

import os
import numpy as np

# TODO: O(n) symmetry

# Degree of freedom: R^N
N = 1

scal_precision = os.environ.get('PRECISION', 'double')

if scal_precision == 'single':
    print("Using single precision")
    SCAL_TYPE = np.float32
    SCAL_TYPE_REAL = np.float32
    LATT_TYPE = np.int32

elif scal_precision == 'double':
    print("Using double precision")
    SCAL_TYPE = np.float64
    SCAL_TYPE_REAL = np.float64
    LATT_TYPE = np.int32
    
else:
    print("Unsupported precision: " + scal_precision)