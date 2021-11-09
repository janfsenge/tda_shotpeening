# auxiliary.py

"""Defines several functions making checks and helper functions."""

# %%

from pathlib import Path, PurePath
import pandas as pd
import numpy as np

# %%
# figures
    
def savefig_severalformats(fig, filename, 
                           path, 
                           format=['pdf', 'jpg']):
    
    if isinstance(path, PurePath):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        print('Wrong filepath, needs to be String or PosixPath!')
        return None
        
    for filetype in format:
        fn = filename+'.'+filetype
        fig.savefig(path / fn, 
                    dpi=100, bbox_inches='tight')


def check_load_file(filepath):
    """Check if the file in filename exists, if the filepath 
    end with either csv, npy or npz load it."""
    
    # check filepath
    if isinstance(filepath, PurePath):
        pass
    elif isinstance(filepath, str):
        filepath = Path(filepath)
    else:
        print('Wrong filepath, needs to be String or PosixPath!')
        return None
    
    # check for fileextension
    if filepath.suffix == '.csv':
        return (pd.read_csv(filepath))
    elif filepath.suffix in ['.npz', '.npy']:
        return (np.load(filepath))
    else:
        print('Wrong file extension! needs to be csv, npz, npy.')
        return None

