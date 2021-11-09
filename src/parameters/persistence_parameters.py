
# %%
# import packages

import os.path
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print(os.getcwd())

import numpy as np

import gtda
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, PersistenceEntropy, Amplitude, NumberOfPoints
import gudhi as gd
# from gudhi.representations import SlicedWassersteinDistance
from gudhi.representations import Entropy

from itertools import chain
# %%
# auxiliary functions for the persistence diagrams and new ones;
# are not used for the results - vector distance is not worth it

def giotto2gudhi(diags, dim=None, np_array=False):
    """Transform giotto 3-dim np.ndarray to list of (list of) tuples."""
    if diags.ndim == 2:
        diags = [diags]
    if np.shape(diags)[-1] == 3:
        diags_list = []
        if dim is not None:
            if np_array:
                for diag in diags:
                    tmp = np.array([[diag[i, 0], diag[i, 1]]
                                    for i in range(np.shape(diag)[0])
                                    if diag[i, 0] != diag[i, 1] and int(diag[i, 2]) == dim])
                    diags_list.append(tmp)
            else:
                for diag in diags:
                    tmp = [(int(diag[i, 2]), (diag[i, 0], diag[i, 1]))
                           for i in range(np.shape(diag)[0])
                           if diag[i, 0] != diag[i, 1] and int(diag[i, 2]) == dim]
                    diags_list.append(tmp)
        else:
            print('no dim')
            for diag in diags:
                tmp = [(int(diag[i, 2]), (diag[i, 0], diag[i, 1]))
                       for i in range(np.shape(diag)[0])
                       if diag[i, 0] != diag[i, 1]]
                diags_list.append(tmp)

        return (diags_list)
    else:
        diags_list = []
        if np_array:
            for diag in diags:
                tmp = np.array([[diag[i, 0], diag[i, 1]]
                                for i in range(np.shape(diag)[0])
                                if diag[i, 0] != diag[i, 1]])
            else:
                for diag in diags:
                    tmp = [(int(diag[i, 2]), (diag[i, 0], diag[i, 1]))
                           for i in range(np.shape(diag)[0])
                           if diag[i, 0] != diag[i, 1]]
                    diags_list.append(tmp)
        return ([(diags[i, 0], diags[i, 1]) for i in range(np.shape(diags)[0])])


def euclidean_dist_parameters(df, df2, cols):
    """helper fucntion to calculate eucldiean distance between param values"""
    return (np.abs(np.array([df[col].values.reshape(-1, 1) - df2[col].values.reshape(1, -1)
                             for col in cols])))


def persistence_vector_amplitude(arr_0, dims=None):
    """ Calculate the vector persistence distance to the empty Persistence Diagram."""
    # if we have several persistence diagrams to consider
    if arr_0.ndim == 3:
        # get the right format for the dimensions present
        if dims is None:
            dims = np.unique(arr_0[:, :, 2])
        elif isinstance(dims, int):
            dims = [dims]

        # calculate the vector-amplitude for each PD in the array and dimension
        result = np.array([[np.linalg.norm(arr_0[i, arr_0[i, :, 2] == dim, 1]
                                           - arr_0[i, arr_0[i, :, 2] == dim, 0])
                            if (arr_0[i, :, 2] == dim).any()
                            else np.nan
                           for dim in dims]
                          for i in range(np.shape(arr_0)[0])]
                          )
    else:
        if dims is None:
            dims = np.unique(arr_0[:, 2])
        elif isinstance(dims, int):
            dims = [dims]

        # calculate the vector_amplitude for each dimension
        result = np.array([np.linalg.norm(arr_0[arr_0[:, 2] == dim, 1]
                                          - arr_0[arr_0[:, 2] == dim, 0])
                           if (arr_0[:, 2] == dim).any()
                           else np.nan
                          for dim in dims]
                          )

    if np.shape(result)[1] == 1:
        result = result.reshape(-1)
    return (result)


def persistence_vector_distance(arr_0, arr_1=None,
                                dim=None, length=None,
                                return_vector=False):
    """
    Calculate the vector-persistence distance.

    Parameters
    ----------
    arr_0 : TYPE
        DESCRIPTION.
    arr_1 : TYPE
        DESCRIPTION.
    dim : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # if we only provide one array, calculate its distance to the 0 vector.
    if arr_1 is None:
        arr_1 = np.zeros([np.shape(arr_0)[0], 2])

    # check dimensions
    assert (np.shape(arr_0)[1] == 2) or (np.shape(arr_0)[1] == 3)
    assert (np.shape(arr_1)[1] == 2) or (np.shape(arr_1)[1] == 3)

    # pick the dimensions used
    if dim is not None:
        assert isinstance(dim, int)

        if np.shape(arr_0)[1] == 3:
            arr_0 = arr_0[arr_0[:, 2] == dim]
        if np.shape(arr_1)[1] == 3:
            arr_1 = arr_1[arr_1[:, 2] == dim]

    # Get the persistence and sort it,
    arr_0 = np.abs(arr_0[:, 1] - arr_0[:, 0])
    arr_0 = arr_0[np.argsort(arr_0)]

    arr_1 = np.abs(arr_1[:, 1] - arr_1[:, 0])
    arr_1 = arr_1[np.argsort(arr_1)]

    # if length parameter is provided only use the first n='length'
    # persistence values
    if length is not None:
        arr_0 = arr_0[:length]
        arr_1 = arr_1[:length]

    # make arrays comparable, pad with zeros if necessary
    tmp = np.size(arr_0) - np.size(arr_1)
    if tmp > 0:  # arr_0 is longer than arr_1
        arr_1 = np.pad(arr_1, (0, tmp), mode='constant')
    elif tmp < 0:
        arr_0 = np.pad(arr_0, (0, -tmp), mode='constant')

    # if return_vector is True, also return the persistence vectors
    if return_vector:
        return([arr_0, arr_1, np.linalg.norm(arr_0 - arr_1)])
    else:
        return(np.linalg.norm(arr_0 - arr_1))


def pairwise_vector_distance(arrays_0, arrays_1=None, dim=0):
    # arrays_0 = columns
    # arrays_1 = rows
    n = np.shape(arrays_0)[0]
    if arrays_1 is not None:
        m = np.shape(arrays_1)[0]
        distance = np.zeros([m, n])

        for i in range(m):
            for j in range(n):
                distance[i, j] = persistence_vector_distance(arrays_1[i],
                                                             arrays_0[j],
                                                             dim=dim)
    else:
        distance = np.zeros([n, n])
        for i in range(n):
            for j in range(i+1, n):
                distance[i, j] = persistence_vector_distance(arrays_0[i],
                                                             arrays_0[j],
                                                             dim=dim)
        distance += distance.transpose()

    return (distance)


def take_dim_from_pd(diagrams, dim=0):
    return(np.array([diagrams[i, diagrams[i, :, 2] == dim, :]
                     for i in range(np.shape(diagrams)[0])]))
    
# %%
# Calculate all persistence diagrams

# TODO do a normal version as well

def getPersistence(data, reduced_homology=False, coeff=2, 
                   infinite_value='max_value'):
    infinite_value = np.max([np.max(data)+1, 2])
    cc = CubicalPersistence(reduced_homology=reduced_homology, coeff=coeff,
                            infinity_values=infinite_value,
                            n_jobs=-1)
    pers = cc.fit_transform(data)
    
    # DIFFERENCE TO NORMAL ALGORITHMS!
    # check for the infinite values and set them to the maximal value of the
    # corresponding heightmap
    for pers_i in range(np.shape(pers)[0]):
        index_max = np.where(pers[0, :, 1] == np.max(pers[0, :, 1]))[0]
        pers[pers_i, index_max, 1] = np.max(data[pers_i])
    
    return pers
   
# TODO rewrite wrapper for computing just some parameters
def persistence_parameters(pers):
    """Compute persistence diagrams as well as scalar persistence-based parameters.

    Parameters
    ----------
    data : np.array of dimension mxm
        matrices of dimension mxm
    reduced_homology : bool, optional
        [description], by default False
    coeff : int, optional
        [description], by default 2
    """
    
    # get the names for the tda parameters
    tda_names = ['numPts_0', 'numPts_1',
                 'entropy_giotto_0', 'entropy_giotto_1',
                 'entropy_giotto_norm_0', 'entropy_giotto_norm_1',
                 'entropy_gudhi_0', 'entropy_gudhi_1',
                 'entropy_gudhi_norm_0', 'entropy_gudhi_norm_1',
                 'mean_pers_0', 'std_pers_0',
                 'mean_pers_1', 'std_pers_1',
                 'norm_pers_0', 'norm_pers_1',
                 'max_pers_0', 'max_pers_1'#,
                #  'carlsson_1', 'carlsson_2', 
                #  'carlsson_3', 'carlsson_4'
                 ]

    amplitude = ['bottleneck', 'wasserstein', 'betti', 'landscape',
                 'silhouette', 'heat', 'persistence_image']
    amplitude_names = list(chain.from_iterable((f'amplitude_{amp}_0', f'amplitude_{amp}_1') for amp in amplitude))
    
    # add the amplitude names as well as vector-persistence to the columns names
    tda_names += amplitude_names + ['amplitude_vector_0', 'amplitude_vector_1']
    tda_array = np.zeros([np.shape(pers)[0], len(tda_names)])

    # number of points in diagram
    numpts = NumberOfPoints(n_jobs=-1).fit_transform(pers)
    tda_array[:, 0:2] = numpts

    # giotto - scalar entropy
    entropy = PersistenceEntropy(
        normalize=False, n_jobs=-1).fit_transform(pers)
    tda_array[:, 2:4] = entropy

    entropy = PersistenceEntropy(normalize=True, n_jobs=-1).fit_transform(pers)
    tda_array[:, 4:6] = entropy

    # gudhi - entropy
    ent = Entropy(mode='scalar', normalized=False)
    tda_array[:, 6] = ent.fit_transform(
        giotto2gudhi(pers, 0, np_array=True)).reshape(-1,)
    tda_array[:, 7] = ent.fit_transform(
        giotto2gudhi(pers, 1, np_array=True)).reshape(-1,)

    ent = Entropy(mode='scalar', normalized=True)
    tda_array[:, 8] = ent.fit_transform(
        giotto2gudhi(pers, 0, np_array=True)).reshape(-1,)
    tda_array[:, 9] = ent.fit_transform(
        giotto2gudhi(pers, 1, np_array=True)).reshape(-1,)

    # calculate statistics on the persistence pairs itself
    bars = []
    for i in range(np.shape(pers)[0]):
        perslen0 = (pers[i, pers[i][:, 2] == 0, 1] -
                    pers[i, pers[i][:, 2] == 0, 0])
        perslen1 = (pers[i, pers[i][:, 2] == 1, 1] -
                    pers[i, pers[i][:, 2] == 1, 0])

        bars.append([np.mean(perslen0[perslen0 > 0]), 
                     np.std(perslen0[perslen0 > 0]),
                     np.mean(perslen1[perslen1 > 0]), 
                     np.std(perslen1[perslen1 > 0]),
                     np.linalg.norm(perslen0), 
                     np.linalg.norm(perslen1),
                     np.max(perslen0), 
                     np.max(perslen1)
                     ])

    tda_array[:, 10:18] = np.array(bars)
    
    # calculate the "carlsson" 4:
    # sum b_i (di- bi)
    

    # Amplitudes
    # starting from count, calculate the amplitudes
    count = 18
    for dist in amplitude:
        amp = Amplitude(metric=dist, n_jobs=-1).fit_transform(pers)
        tda_array[:, count:count+2] = amp
        count += 2

    # Get the persistence_vector distance
    tda_array[:, count:count +
              2] = persistence_vector_amplitude(pers, dims=[0, 1])

    # join together the arrays and return
    return([tda_names, tda_array])