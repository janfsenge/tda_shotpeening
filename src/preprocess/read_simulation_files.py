""" 
Read in either file from csv files (for the numerical simulation) 
or random regions of different sizes or one size for a larger filesize (for experimental data).

Numerical simulations have slighlty different grids as well as slighlty uneven grid spacing. 
Hence, we need to calculate the vheight values on a reagular grid which is the same for all samples. 

To prevent inaccuries when using a regular grid of the same size as the grids provided, 
the grid size is increased for a multiple of the size of the original grid.

We use both RegularInterpolateGrid as well as Griddata. The first one should be more precise (?)


#TODO: properly comment
"""

# %%
# Load the modules
import time
import numpy as np
import pandas as pd

from scipy.interpolate import griddata, RegularGridInterpolator

# only works (without install pathlib) for python 3.4+
from pathlib import Path, PurePath


# %%
# define the two functions of this module
def check_filename(file):
    """Check if the str in file has the proper form."""
    filename = file.name
    if (isinstance(filename, str) 
        and filename.startswith("heightmap_sequence-")
        and filename.endswith(".csv")
        and len(filename.split('_'))==4
        and filename.split('_')[2].startswith("impacts-")
        and filename.split('_')[3].startswith("coverage-")
    ):
        return True
    
    else:
        return False


def read_heightmaps_sequences(folderpath,
                              z_name='U3',
                              x_name='x', 
                              y_name='y'):
    """read the files and save them with their grids in the lists z, x, y;
    df_data save the meta information obtained from the filenamse.

    Parameters
    ----------
    folder_name : str
        Path to the folder where the numerical simulation csv are
    z_name : str, optional
        column name of the values where the height values are, by default 'U3'
    x_name : str, optional
        column names of the x-values, by default 'x'
    y_name : str, optional
        column names of the y-values, by default 'y'
    """
    df_data = []
    np_data = []
    count = 0
    
    
    if isinstance(folderpath, str):
        folderpath = Path(folderpath)
    
    if isinstance(folderpath, PurePath):
        for file in folderpath.iterdir():
            if check_filename(file):
                sequence = file.name.split('_')[1].split('-')[1]
                impacts = file.name.split('_')[2].split('-')[1]
                coverage = file.name.split('_')[-1].split('-')[1].split('.')[0]
            
                df_tmp = pd.read_csv(file)
                df_tmp = df_tmp.rename(columns={col: col.strip()
                                                for col in df_tmp.columns})
                # sort by y, x to keep consistencey with np.meshgrid ij style
                df_tmp = df_tmp.sort_values(by=[y_name, x_name])

                size = [len(np.unique(df_tmp[x_name])),
                        len(np.unique(df_tmp[y_name]))]

                df_data.append({'id': count,
                                'sequence': int(sequence),
                                'impacts': int(impacts),
                                'coverage': int(coverage)
                                })
                # order x, y, z
                np_data.append([df_tmp[x_name].values.reshape(size),
                                df_tmp[y_name].values.reshape(size),
                                df_tmp[z_name].values.reshape(size)
                                ])
                count += 1

        np_data = np.array(np_data)
        df_data = pd.DataFrame(df_data)

        df_data = df_data.sort_values(by=['sequence', 'impacts'])
        # pick only those having impacts below or equal to 100% (if they exist)
        df_data = df_data[df_data['impacts'] < 35].reset_index(drop=True)

        # order the heightmaps in the numpy array according to the new
        # ordering in the dataframe
        np_data = np_data[df_data['id'].values]
        # reset the count/listpositon variable
        df_data.loc[:, 'id'] = df_data.index

        # x, y, z, df_data
        return(np_data[:, 2], np_data[:, 0], np_data[:, 1], df_data)
    
    print('Folderpath needs to be PurePath or string.')
    return None

def check_if_regulargrid(x, y):
    """Check if x, y span a regular grid (with uneven spacing allowed).
    
    Return if it's a regular grid.
    """
    assert np.shape(x) == np.shape(y)

    checks = []
    # check if values in each row/column are the same
    tmp = [x[i, k] == x[i, 0]
           for i in range(x.shape[0]) for k in range(x.shape[1])]
    checks.append(np.array(tmp).all())
    tmp = [x[i, k] == x[0, k]
           for i in range(x.shape[0]) for k in range(x.shape[1])]
    checks.append(np.array(tmp).all())

    tmp = [y[i, k] == y[0, k]
           for i in range(x.shape[0]) for k in range(x.shape[1])]
    checks.append(np.array(tmp).all())
    tmp = [y[i, k] == y[i, 0]
           for i in range(x.shape[0]) for k in range(x.shape[1])]
    checks.append(np.array(tmp).all())

    if (checks[0] & checks[2] & (not checks[1]) & (not checks[3])):
        return True
    elif (checks[1] & checks[3] & (not checks[0]) & (not checks[2])):
        return True
    elif checks.all():
        # not even a grid
        return False
    else:
        return False

def interpolate_regular_grid(z, x, y,
                             interpolate_size,
                             bigger_grid=False):
    """For a list of values z,x,y interpolate the points to a regular grid using interpolate_size.

    Doesn't assume that a regular grid is present. 

    Parameters
    ----------
    z : [type]
        [description]
    x : list of x_grids
        [description]
    y : [type]
        [description]
    interpolate_size : [type]
        [description]
    bigger_grid: boolean
        Trigger if we are creating a new grid using the maximal grid borders or 
        if we take the smallest grid border over all grids present when
        constructing the new grids.
    """
    assert x.ndim == y.ndim and z.ndim == x.ndim
    if x.ndim == 3:
        loop = range(x.shape[0])
    else:
        loop = [0]
        x = np.array([x])
        y = np.array([y])
        z = np.array([z])

    # check if all grids are the same
    if (np.array([xi == x[0] for xi in x]).all()
            and np.array([xi == x[0] for xi in x]).all()):
        # doesn't change anything, but we can save some time
        bigger_grid = True

    # check if grids are regular

    if np.array(interpolate_size).size == 2:
        interpolate_x = interpolate_size[0]
        interpolate_y = interpolate_size[1]
    else:
        interpolate_x = interpolate_size
        interpolate_y = interpolate_size

    if bigger_grid:
        xmax = np.max(x)
        xmin = np.min(x)

        ymax = np.max(y)
        ymin = np.min(y)
    else:
        # Take the smallest maximal value and largest minimal value to be endpoints
        # of the new grid
        xmax = x.max(axis=1).max(axis=1).min()
        xmin = x.min(axis=1).min(axis=1).max()

        ymax = y.max(axis=1).max(axis=1).min()
        ymin = y.min(axis=1).min(axis=1).max()

    x_new, y_new = np.meshgrid(np.linspace(xmin, xmax, interpolate_x),
                               np.linspace(ymin, ymax, interpolate_y))
    
    # use fortran ordering so that when doing reshpae later we get the same 
    # ordering as for the x/y-meshgrids
    xy_list = np.reshape((x_new, y_new), (2, -1)).T
    
    z_new = np.zeros([np.shape(x)[0], np.shape(x_new)[0], np.shape(x_new)[1]])
    for i in loop:
        # using RegularGridInterpolator points outside the original grid
        # are extrapolated; so we can always use the larger grid if we want
        reg_inter = RegularGridInterpolator(points=(np.unique(x[i]),
                                                    np.unique(y[i])),
                                            values=z[i],
                                            method='linear',
                                            bounds_error=False, # extrapolate the data outside the bounds
                                            fill_value=None) # it seems it uses nearest neighbour extrapolation?
        
        z_new[i] = reg_inter(xy_list).reshape([interpolate_x, interpolate_y],
                                              order='F')

    return (z_new, x_new, y_new)


def interpolate_grid(z, x, y,
                     interpolate_size,
                     bigger_grid=False):
    """For a list of values z,x,y interpolate the points to a regular grid using interpolate_size.

    Doesn't assume that a regular grid is present. 

    Parameters
    ----------
    z : [type]
        [description]
    x : [type]
        [description]
    y : [type]
        [description]
    interpolate_size : [type]
        [description]
    bigger_grid: boolean
        Trigger if we are creating a new grid using the maximal grid borders or 
        if we take the smallest grid border over all grids present when
        constructing the new grids.
    """
    assert x.ndim == y.ndim and z.ndim == x.ndim
    if x.ndim == 3:
        loop = range(x.shape[0])
    else:
        loop = [0]
        x = np.array([x])
        y = np.array([y])
        z = np.array([z])

    # check if all grids are the same
    if (np.array([xi == x[0] for xi in x]).all()
            and np.array([xi == x[0] for xi in x]).all()):
        # doesn't change anything, but we can save some time
        bigger_grid = True

    # check if grids are regular

    if np.array(interpolate_size).size == 2:
        interpolate_x = interpolate_size[0]
        interpolate_y = interpolate_size[1]
    else:
        interpolate_x = interpolate_size
        interpolate_y = interpolate_size

    if bigger_grid:
        xmax = np.max(x)
        xmin = np.min(x)

        ymax = np.max(y)
        ymin = np.min(y)
    else:
        # Take the smallest maximal value and largest minimal value to be endpoints
        # of the new grid
        xmax = x.max(axis=1).max(axis=1).min()
        xmin = x.min(axis=1).min(axis=1).max()

        ymax = y.max(axis=1).max(axis=1).min()
        ymin = y.min(axis=1).min(axis=1).max()

    x_new, y_new = np.meshgrid(np.linspace(xmin, xmax, interpolate_x),
                               np.linspace(ymin, ymax, interpolate_y))
    z_new = np.zeros([np.shape(x)[0], np.shape(x_new)[0], np.shape(x_new)[1]])
    for i in loop:
        # picking the minimum and maximum over all will not give values in -0.1 but
        # -0.99999999999 up to ...
        # we could go for the larger variant, but then we would be needing to
        if bigger_grid:
            z_new[i] = griddata(points=(x[i].flatten(), y[i].flatten()),
                                values=z[i].flatten(),
                                xi=(x_new, y_new),
                                method='linear',
                                fill_value=np.nan)

            # For the values outside of the convex hull,
            # use the nearest method to get their values
            mask = np.isnan(z_new[i])
            if mask.any():
                z_new[i][mask] = griddata(points=(x[i].flatten(), y[i].flatten()),
                                          values=z[i].flatten(),
                                          xi=(x_new[mask], y_new[mask]),
                                          method='nearest')

        # if taking the smaller grid, we can just apply linear method
        else:
            z_new[i] = griddata(points=(x[i].flatten(), y[i].flatten()),
                                values=z[i].flatten(),
                                xi=(x_new, y_new),
                                method='linear')

    return (z_new, x_new, y_new)
# %%
# Call all the functions with their different parameters for the numerical simulation data.

# TODO change to current folder structure
# if __name__ == '__main__':
    
#     source_path = Path(__file__).parents[2]
#     dir_read_path = Path.joinpath(source_path, 'data',
#                                   'surfaces_real',
#                                   'heightmaps_simulation')
#     dir_save_path = Path.joinpath(source_path, 'data',
#                                   'surfaces_real',
#                                   'heightmaps_simulation')

#     # read the files and save them with their grids in the lists z, x, y
#     # df_data save the meta information obtained from the filenamse
#     z, x, y, df_data = read_heightmaps_sequences(folderpath=dir_read_path,
#                                                  z_name='U3', x_name='x', y_name='y')

#     # Check the grids for each of the grids in the lists
#     check = np.array([x[i] == x[0] for i in range(x.shape[0])])
#     print('Are all x_grids the same?', check.all())
#     check = np.array([y[i] == y[0] for i in range(x.shape[0])])
#     print('Are all y_grids the same?', check.all())

#     # check if we have a regular grid
#     check = np.array([check_if_regulargrid(x[i], y[i])
#                      for i in range(x.shape[0])])
#     print('Do we have regular grids:', check.all())

#     # Just for all purposes (must be one) if the grids are of 
#     # a certain compatible format
#     assert np.array([len(np.unique(np.diff(x[i], axis=0)))
#                     == 1 for i in range(x.shape[0])]).all()
#     assert np.array([len(np.unique(np.diff(y[i], axis=1)))
#                     == 1 for i in range(y.shape[0])]).all()

#     # Check if we have evenly spaced grids
#     check = np.array([len(np.unique(np.diff(x[i], axis=1)))
#                      == 1 for i in range(x.shape[0])])
#     print('Are all x_grids evenely spaced?', check.all())
#     check = np.array([len(np.unique(np.diff(y[i], axis=0)))
#                      == 1 for i in range(y.shape[0])])
#     print('Are all y_grids evenely spaced?', check.all())

#     # Now 
#     t0 = time.time()
#     znew, xnew, ynew = interpolate_grid(
#         z, x, y, 74*14, bigger_grid=False)
#     print(f'No big grid took {time.time()-t0} seconds!')

#     t0 = time.time()
#     znew2, xnew2, ynew2 = interpolate_grid(
#         z, x, y, 74*14, bigger_grid=True)
#     print(f'Big grid took {time.time()-t0} seconds!\n')
    
#     # use regulargrid method
#     t0 = time.time()
#     rznew, rxnew, rynew = interpolate_regular_grid(
#         z, x, y, 74*14, bigger_grid=False)
#     print(f'No big grid took {time.time()-t0} seconds!')

#     t0 = time.time()
#     rznew2, rxnew2, rynew2 = interpolate_regular_grid(
#         z, x, y, 74*14, bigger_grid=True)
#     print(f'Big grid took {time.time()-t0} seconds!\n')
    
#     # check if griddate and regularinterpolator give the same values:
#     # smaller grid:
#     if (znew == rznew).all():
#         print('Griddata and RegularInterpolator give the same on smaller grid.')
#     else:
#         print('Griddata != RegularInterpolator on smaller grid!')
#     # bigger grid:
#     if (znew2 == rznew2).all():
#         print('Griddata and RegularInterpolator give the same on bigger grid.')
#     else:
#         print('Griddata != RegularInterpolator on bigger grid!')
    
#     # Check the differences between the height values calculated by the different
#     # strategies
#     print('The maximal difference between the different heightmaps is', 
#           np.max(np.abs(znew-znew2)))
#     print('The maximal difference between the different heightmaps (GRIDDATA vs RegularGridInterpolator)', 
#           np.max(np.abs(znew-rznew)))
#     print('The maximal difference between the different heightmaps (GRIDDATA vs RegularGridInterpolator)', 
#           np.max(np.abs(znew2-rznew2)))
#     print('The maximal difference between the x_grids is',
#           np.max(np.abs(xnew-xnew2)))
#     print('The maximal difference between the y_grids is',
#         np.max(np.abs(ynew-ynew2)))
