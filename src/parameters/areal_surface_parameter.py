
# %%
from scipy.integrate import trapezoid, simpson
from scipy.stats import kurtosis, skew
import numpy as np
import pandas as pd

# TODO Do a parallel version
# TODO flatten the grids here
# TODO update docstring
# TODO do a scikit version


def getRoughnessParams(z_values,
                       x_grid=None,
                       y_grid=None,
                       version='all',
                       integral='trapezoid',
                       shift=True,
                       dimension=2,
                       return_shift=False):
    """wrapper method for compute_roughnessparameters"""

    if isinstance(z_values, list) or \
            (isinstance(z_values, np.ndarray) and z_values.ndim == dimension+1):

        df_data = []
        for z in z_values:
            param, name, name_long = compute_roughnessparameters(z,
                                                                 x_grid, y_grid,
                                                                 version, integral,
                                                                 shift, return_shift)
            df_data.append({nstr: param[ni] for ni, nstr in enumerate(name)})

        return (pd.DataFrame(df_data))

    else:
        return(compute_roughnessparameters(z_values,
                                           x_grid, y_grid,
                                           version, integral,
                                           shift, return_shift))



def compute_roughnessparameters(z_values,
                                x_grid=None,
                                y_grid=None,
                                version='all',
                                integral='trapezoid',
                                shift=True,
                                return_shift=False):
    """Calculate different surface roughness parameters.

    Height values are in relation to a reference plane. The z_values is given in the form
    of height deviations from the normal material.
    Thus we characterize the shot peening process and not the material per se.

    Normalization is therefore not done.

    Parameters
    ----------
    z_values : TYPE
        DESCRIPTION.
    version : int or str, optional
        Sets the version of the parameters.
        The default is 0.
    shift : boolean, optional
        If True substract the reference plane from the z_values.
        The default is False.
    *kwargs : TYPE
        evaluationsize : int
            Number of simiilar sized evaluation length / size.

    Returns
    -------
    param : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    name_long : TYPE
        DESCRIPTION.

    """
    if x_grid is not None and y_grid is not None:
        x_spread=(np.max(x_grid) - np.min(x_grid))
        y_spread=(np.max(y_grid) - np.min(y_grid))
        grid_area=x_spread * y_spread
    else:
        grid_area=1
        integral='absolute'

    z_original = np.copy(z_values)
    if shift:
        if integral == 'trapezoid':
            x_grid=np.unique(x_grid).flatten()
            y_grid=np.unique(y_grid).flatten()
            z_values -= trapezoid([trapezoid(z_val_x, x_grid)
                                   for z_val_x in z_values], y_grid) / grid_area
        elif integral == 'simpson':
            x_grid=np.unique(x_grid).flatten()
            y_grid=np.unique(y_grid).flatten()
            z_values -= simpson([simpson(z_val_x, x_grid)
                                 for z_val_x in z_values], y_grid) / grid_area
        else:
            z_values -= np.mean(z_values)

    name=[]
    name_long=[]
    param=[]

    sq=None
    # Amplitude parameters
    # statistical height descriptors
    if version == 'all' or version == 0 or str(version).lower() in ['sa']:
        name.append('Sa')
        name_long.append('arithmetic mean')

        if integral == 'trapezoid':
            param.append(trapezoid([trapezoid(np.abs(z_val_x), x_grid)
                                    for z_val_x in z_values], y_grid) / grid_area)
        elif integral == 'simpson':
            param.append(simpson([simpson(np.abs(z_val_x), x_grid)
                                  for z_val_x in z_values], y_grid) / grid_area)
        else:
            param.append(np.mean(np.abs(z_values)))

    if version == 'all' or version == 1 or str(version).lower() in ['sq']:
        name.append('Sq')
        name_long.append('root mean square')

        if integral == 'trapezoid':
            sq=np.sqrt(trapezoid([trapezoid(z_val_x**2, x_grid)
                                    for z_val_x in z_values], y_grid) / grid_area)
        elif integral == 'simpson':
            sq=np.sqrt(simpson([simpson(z_val_x**2, x_grid)
                                  for z_val_x in z_values], y_grid) / grid_area)
        else:
            sq=np.sqrt(np.mean(z_values**2))
        param.append(sq)

    if version == 'all' or version == 2 or str(version).lower() in ['ssk', 'skew']:
        name.append('Ssk')
        name_long.append('skewness')

        if sq is None:
            if integral == 'trapezoid':
                sq=np.sqrt(trapezoid([trapezoid(z_val_x**2, x_grid)
                                        for z_val_x in z_values], y_grid) / grid_area)
            elif integral == 'simpson':
                print('This shouldnt show')
                sq=np.sqrt(simpson([simpson(z_val_x**2, x_grid)
                                      for z_val_x in z_values], y_grid) / grid_area)

            else:
                print('Error')

        if integral == 'trapezoid':
            param.append((trapezoid([trapezoid(z_val_x**3, x_grid)
                                     for z_val_x in z_values], y_grid) / grid_area) / (sq**3))
        elif integral == 'simpson':
            param.append((simpson([simpson(z_val_x**3, x_grid)
                                   for z_val_x in z_values], y_grid) / grid_area) / (sq**3))
        else:
            # biased skewness
            param.append(skew(z_values.ravel()))

    if version == 'all' or version == 3 or str(version).lower() in ['sku', 'kurtosis']:
        name.append('Sku')
        name_long.append('kurtosis')

        if sq is None:
            if integral == 'trapezoid':
                sq=np.sqrt(trapezoid([trapezoid(z_val_x**2, x_grid)
                                        for z_val_x in z_values], y_grid) / grid_area)
            elif integral == 'simpson':
                sq=np.sqrt(simpson([simpson(z_val_x**2, x_grid)
                                      for z_val_x in z_values], y_grid) / grid_area)
            else:
                print('Error')

        if integral == 'trapezoid':
            param.append((trapezoid([trapezoid(z_val_x**4, x_grid)
                                     for z_val_x in z_values], y_grid) / grid_area) / (sq**4))
        elif integral == 'simpson':
            param.append((simpson([simpson(z_val_x**4, x_grid)
                                   for z_val_x in z_values], y_grid) / grid_area) / (sq**4))
        else:
            # biased Pearson kurtosis.
            param.append(kurtosis(z_values.ravel(), fisher=False))

    # extreme-value height descriptors
    if version == 'all' or version == 4 or str(version).lower() in ['sz']:
        name.append('Sz')  # total roughness R_t
        name_long.append('maximal peak to valley difference')
        # with initial we ignore the values below or above 0-value.
        param.append(np.max(z_values, initial=0) +
                     np.abs(np.min(z_values, initial=0)))

    # elif version == 5 or str(version).lower() in ['szavg']:
    #     # evaluationsize = samplesize
    #     if 'evaluationsize' in kwargs:
    #         size = kwargs.get("evaluationsize")
    #     else:
    #         size = 5
    #     name = 'Sz'+str(size**2)
    #     name_long = str(size**2)+' point peak-valley-height'
    #     param = evaluation_heights(z_values, size=size)

    # # Material Ratio Parameters
    # Smr, Smc, Sxp are functions, add later
    # # Functional parameters

    # # Hybriid parameters
    # TODO do integral versions
    if version == 'all' or version == 5 or str(version).lower() in ['Sdq']:
        name.append('Sdq')
        name_long.append('root mean square gradient')
        tmp = np.gradient(z_original, x_grid, y_grid)

        # need to account for the grid_sizes
        if integral == 'trapezoid':
            param.append(np.sqrt(np.mean(tmp[0]**2 + tmp[1]**2)))
        elif integral == 'simpson':
            param.append(np.sqrt(np.mean(tmp[0]**2 + tmp[1]**2)))
        else:
            param.append(np.sqrt(np.mean(tmp[0]**2 + tmp[1]**2)))

    # percentage. The higher, the complexer the surface.
    # (greatly influenced by number of points?)

    if version == 'all' or version == 6 or str(version).lower() in ['Sdr']:
        name.append('Sdr')
        name_long.append('developed interfacial area ratio')
        
        # using numpy gradient
        tmp = np.gradient(z_original, x_grid, y_grid)
        param.append(np.mean(np.sqrt(1+tmp[0]**2 + tmp[1]**2)-1))

        # # using manual calculation:
        # dz_dx = np.diff(z_original) / np.diff(x_grid)
        # dz_dy = (np.diff(z_original.T) / np.diff(y_grid)).T

        # idx = [np.min(np.shape(dz_dx)), np.min(np.shape(dz_dy))]

        # Sdr = np.sum(np.sqrt(1+np.square(dz_dx[:idx[0], :idx[1]])
        #                     +np.square(dz_dy[:idx[0], :idx[1]]))
        #             -1)/(idx[0]*idx[1])
        # param.append(Sdr)

    if return_shift:
        return (z_values, param, name, name_long)
    else:
        return (param, name, name_long)


if __name__ == '__main__':
    data_simu=np.load('../data/heightmaps/simulationHeightmaps.npz')

    i=0
    z=data_simu['values'][i]
    x=np.unique(data_simu['x_grid']).flatten()  # can also just take column
    y=np.unique(data_simu['y_grid']).flatten()  # can also just take row

    param, name, name_long=getRoughnessParams(z, x, y,
                                                version='all',
                                                shift=True)
    print(param)

# %%
# depends on the path you have opened
