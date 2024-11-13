"""
    Select suitable gauge group.
"""
import os

su_group = os.environ.get('GAUGE_GROUP', 'su2').lower()

if su_group == 'su2':
    print("Using SU(2)")
    from .su2 import *
elif su_group == 'sl2c':
    print("Using SL(2,C)")
    from .sl2c import *
else:
    print("Unsupported gauge group: " + su_group)
    exit()