"""
Implement methods to find local mnima and calculate coverage values
according to the definition of coverage percentage calculation 
used in the FE coverage eveolution model.
"""

# %%
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

# from skimage import data, img_as_float

# %%
# size of the indent circles and the larger circle


def get_local_minima(z, grid,
                     radius_indent, 
                     eps, 
                     df_info):
    """calculate the local minima from images in z.

    Parameters
    ----------
    z : [n,n] nd.array
        height values for which we want to calculate local minima.
    grid : [n,n] nd.array
        any kind of grid; x_gird or y_grid. 
        values should be repeated in one direction.
    eps : 
    df_info : dataframe
        contains values for position as 'id' and 
        number of 'impacts' for the surfaces
    """
    # check dataframe:
    assert 'id' in df_info.columns
    assert 'impacts' in df_info.columns
    
    # make sure that we get one difference; so it doesn;t matter
    # if we pass xgrid or ygrid
    griddiff = grid[1, 1] - grid[0, 0]

    coord_list = []
    for id in np.unique(df_info['id']):
        # swap sign, since we want local minima not maxima
        im = -z[id]

        # calculate some distance
        xdist = int(np.ceil((radius_indent-(eps/2))/griddiff))
        coordinates = peak_local_max(im,
                                     min_distance=xdist//3,
                                     exclude_border=False,
                                     num_peaks=df_info.loc[df_info['id'] == id,
                                                           'impacts'].values[0])
        coord_list.append(coordinates)

    # create one numpy array for the coordinates of the minima
    # find the longest array in the list
    max_len = np.max([np.shape(arr)[0] for arr in coord_list])
    minima_arr = np.full((len(coord_list), max_len, 2),
                         fill_value=np.nan)

    # set the values
    for i in range(len(coord_list)):
        minima_arr[i][:coord_list[i].shape[0], :] = coord_list[i]

    return minima_arr
# %%


def approximate_coverage(minima_arr, 
                         xgrid, 
                         ygrid,
                         radius_circ, 
                         radius_indent,
                         df_info):

    # check xgrid and ygrid that they have the right meshgrid format
    assert len(np.unique(xgrid[:, 0])) == 1
    assert len(np.unique(ygrid[0, :])) == 1
    # check that 'id' is a column in df_info
    assert 'id' in df_info.columns

    # get the mask for the large circle
    mask_circ = (xgrid**2 + ygrid**2 <= radius_circ**2)

    coverage_list = []
    mask_list = []

    for id in df_info['id']:
        # get the list of min/max points
        pts_list = minima_arr[id]
        pts_list = pts_list[~np.isnan(pts_list[:, 0]), :].astype('int')

        # around each local minima draw a mask with diameter 0.055 and save as elements in an np.array
        mask_pts = np.array([(xgrid[0, pt[0]] - xgrid)**2 + (ygrid[pt[1], 0] - ygrid)**2 <= (radius_indent)**2
                            for pt in pts_list])

        # join the masks in the mask_pts array; use and OR since the circles are disjoint most of the time
        mask_several = np.logical_or.reduce(mask_pts)

        # find the intersection of the uniopn of smaller circles and the larger one
        # need to transpose to get the same x/y direction
        mask = np.logical_and(mask_circ, mask_several).T

        coverage_list.append(np.sum(mask)/np.sum(mask_circ))
        mask_list.append(mask)

    return (coverage_list, mask_list)
