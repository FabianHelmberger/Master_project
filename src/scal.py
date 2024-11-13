"""
    Select suitable scalar type
"""
import os

scal_type = os.environ.get('SCAL_TYPE', 'complex').lower()

if scal_type == 'real':
    from .rn_scal import *
    print(f"Using R^{N}")

elif scal_type == 'complex':
    from .cn_scal import *
    print(f"Using C^{N}")
    
else:
    print("Unsupported scalar type: " + scal_type)
    exit()