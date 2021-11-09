
# %%
import sys
import os.path
from sklearn.base import BaseEstimator, TransformerMixin  # PredictorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import numpy as np
import scipy
import scipy.signal

from sklearn.linear_model import LinearRegression


def wrapper_postprocess(z_values, x_grid, y_grid,
                        removeplane=True,
                        use_filter=True,
                        interpolation_size=1000,
                        cut_matrix=2,
                        dimension=2, cut_off=0.8):
    if isinstance(z_values, list) or \
            (isinstance(z_values, np.ndarray) and z_values.ndim == dimension+1):

        assert x_grid.ndim == dimension
        assert y_grid.ndim == dimension

        znew = []
        for z in z_values:
            zt, xt, yt = postprocess(z, x_grid, y_grid,
                                     removeplane, use_filter,
                                     interpolation_size, cut_matrix, cut_off)
            znew.append(zt)
            
        assert np.array([np.shape(z)==np.shape(znew[0]) for z in znew]).all()
        znew = np.array(znew)

        return (znew, xt, yt)

    else:
        return(postprocess(z_values, x_grid, y_grid,
                           removeplane, use_filter,
                           interpolation_size, cut_matrix, cut_off))


def remove_plane(z, x, y):
    """Remove linear plane from z values. All need to have same shape!

    Parameters
    ----------
    z : [type]
        [description]
    x : [type]
        [description]
    y : [type]
        [description]
    """
    X_reg = np.vstack([x.flatten(), y.flatten(), x.flatten()*y.flatten()]).T
    reg = LinearRegression().fit(X_reg, z.flatten())

    return((z.flatten() - reg.predict(X_reg)).reshape(np.shape(z)))


def postprocess(z, x, y,
                removeplane=True,
                use_filter=True,
                interpolation_size=1000,
                cut_matrix=2,
                cut_off=0.8):
    """Do postprocessing of given square surface heightmap.

    According to ISO 
        1) remove the tilt for the surface by fitting a plane
        2) linearly inteprolate samples for better results filters
        3) do a gaussian filter operation
        4) subtract gaussian filtered surface from surface
        5) cut off some parts at the borders

    Parameters
    ----------
    z : numpy array of size (m,m)
        amplitudes of heightmap
    x_grid : numpy array of size (m,m)
        x-values
    y_grid : numpy array of size (m,m)
        y-values
    interpolation_size : int, optional
        [description], by default 1000

    Returns
    -------
    [type]
        [description]
    """
    if removeplane:
        z = remove_plane(z, x, y)

    if interpolation_size > 0:
        xnew = np.linspace(np.min(x), np.max(x), interpolation_size)
        ynew = np.linspace(np.min(y), np.max(y), interpolation_size)

        xnew_grid, ynew_grid = np.meshgrid(xnew, ynew, indexing='ij')

        xi = np.unique(x)
        yi = np.unique(y)
        height_interpolate = scipy.interpolate.RegularGridInterpolator((xi, yi), z)
        z = height_interpolate((xnew_grid, ynew_grid))

        x = xnew_grid
        y = ynew_grid

    if use_filter:
        # shift the values so that they are positive and start from 0
        # and add a symmetric negative part to it
        x_filter = np.unique(x) - np.min(x)
        x_filter = np.hstack([- np.flip(x_filter[1:]),
                              x_filter])

        y_filter = np.unique(y) - np.min(y)
        y_filter = np.hstack([- np.flip(y_filter[1:]),
                              y_filter])

        # set the cut-off values according to ISO (we are using the same cut-off value)
        # TODO: change cut-off value accoridng to surface/profile

        # since cutoff needs to be the same:
        lambda_c = cut_off

        # 3D GAUSSIAN FILTER APPLIED IN SPACE DOMAIN TO OBTAIN WAVINESS THAT IS
        # LONG WAVELENGTH filter in space domain (Equivalent to LOW-PASS FILTER in
        # frequency domain):

        # alpha only appears in the calculations as square; hence we do not take its
        # square root and square it later.
        alpha2 = np.log(2)/np.pi
        exp_factor = -np.pi / (alpha2 * lambda_c**2)

        # split into filters for x and y
        gauss_filter_y = [np.exp(exp_factor * yfilt_i**2)
                          for yfilt_i in y_filter]
        gauss_filter_x = [np.exp(exp_factor * xfilt_i**2)
                          for xfilt_i in x_filter]
        # get kernel by outer multiplication
        gauss_filter = np.outer(gauss_filter_x, gauss_filter_y)

        # The factor (1/(alpha2 * lambda_c)) will be lost during normalisation
        # gauss_filter = (1/(alpha2 * lambda_c)) * gauss_filter
        # normalization:
        gauss_filter /= np.sum(gauss_filter)

        # 'Convolution scipy fft'
        conv_fft = scipy.signal.fftconvolve(z, gauss_filter, 'same')

        z = z - conv_fft

    if cut_matrix > 0:
        z = z[cut_matrix:-cut_matrix, cut_matrix:-cut_matrix]
        x = x[cut_matrix:-cut_matrix, cut_matrix:-cut_matrix]
        y = y[cut_matrix:-cut_matrix, cut_matrix:-cut_matrix]

    return (z, x, y)


# class fOperator(BaseEstimator, TransformerMixin):
#     """Fitting to a nominal shape.

#     Right now it only fits a plane to the data and removes the tilt
#     by subtracting said plane from the surface.
#     """
#     def __init__(self):
#         pass

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         """Subtract the predicted point of the nominal plane from z."""
#         X, y = check_X_y(X,y)
#         # print('started transform')
#         x_grid = X[:, 0].flatten()
#         y_grid = X[:, 1].flatten()
#         y = y.flatten()
#         X = np.vstack([x_grid, y_grid,
#                        x_grid*y_grid]).T

#         regression = LinearRegression().fit(X, y)
#         X_ = y.flatten() - regression.predict(X)
#         return X_

#     def fit_transform(self, X, y):
#         """Call fit and transform; fit doesn't do anything."""
#         self.fit(X,y)
#         return self.transform(X,y)


# class smooth_interpolate(BaseEstimator, TransformerMixin):
#     """Interpolate the data as a step before doing gaussian smoothing.
#     """
#     def __init__(self, interpolation_size=1000, gridstyle='regular'):
#         self.xnew = None
#         self.ynew = None
#         self.interpolation_size = interpolation_size
#         self.gridstyle = gridstyle

#     def fit(self, X, y=None):
#         """create new grid"""
#         self.xnew = np.linspace(np.min(X[:,0]),
#                                 np.max(X[:,0]),
#                                 self.interpolation_size)
#         self.ynew = np.linspace(np.min(X[:,1]),
#                                 np.max(X[:,1]),
#                                 self.interpolation_size)
#         self.xnew, self.ynew = np.meshgrid(self.xnew,
#                                            self.ynew,
#                                            indexing='ij')
#         return self

#     def transform(self, X, y=None):
#         """Using scipy inteprolation, get linear interp. of values."""
#         # need X and y
#         X, y = check_X_y(X,y)
#         # print('started transform')
#         xi = np.unique(X[:, 0])
#         yi = np.unique(X[:,1])
#         height_interpolate = scipy.interpolate.RegularGridInterpolator((xi, yi), y)
#         znew = height_interpolate((self.xnew, self.ynew))

#         return znew

#     def fit_transform(self, X, y):
#         """Enter ability to use for regressors etc."""
#         self.fit(X,y)
#         return self.transform(X,y)

# %%

if __name__ == '__main__':
    # Run postprocessing for all numerical samples
    filepath = '../../data/surfaces_real/surface_numSimulation.npz'

    if os.path.isfile(filepath):
        data_simu = np.load(filepath)
        print('File found.')
    else:
        filepath = filepath.split('../', 1)[1]
        if os.path.isfile(filepath):
            data_simu = np.load(filepath)
            print('Parent directory used; file found.')
        else:
            print('No file found')
            sys.exit()
            
    x_grid = data_simu['x_grid']
    y_grid = data_simu['y_grid']
    data = data_simu['values']
    
    x = x_grid
    y = y_grid
    z = data[0]

# %%
# Run postprocessing for all numerical samples
if __name__ == '__main__':
    filepath = '../../data/surfaces_real/surface_numSimulation.npz'

    if os.path.isfile(filepath):
        data_simu = np.load(filepath)
        print('File found.')
    else:
        filepath = filepath.split('../', 1)[1]
        if os.path.isfile(filepath):
            data_simu = np.load(filepath)
            print('Parent directory used; file found.')
        else:
            print('No file found')
            sys.exit()
            
    x_grid = data_simu['x_grid']
    y_grid = data_simu['y_grid']
    data = data_simu['values']

    x_arr = []
    y_arr = []
    z_arr = []
    for i in range(data.shape[0]):
        z, x, y = postprocess(data[i], x_grid, y_grid,
                              interpolation_size=1000,
                              cut_matrix=2)
        z_arr.append(z)

        if len(x_arr) == 0:
            x_arr.append(x)
        elif (x != x_arr[0]).any():
            x_arr.append(x)

        if len(y_arr) == 0:
            y_arr.append(y)
        elif (y != y_arr[0]).any():
            y_arr.append(y)

        print(f'Done {i} / {data.shape[0]}.')

    print(f'\nValues: {np.shape(z_arr)}')
    print(f'x_grid: {np.shape(x_arr)}')
    print(f'y_grid: {np.shape(y_arr)}')

    filepath = filepath.split('.npz')[0]
    np.savez_compressed(filepath+'_postproc.npz',
                        values=np.array(z_arr),
                        x_grid=np.array(x_arr),
                        y_grid=np.array(y_arr)
                        )

    print('Calculations done and saved')
# %%
