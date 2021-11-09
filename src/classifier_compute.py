
# %%
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from pathlib import Path

from classifier_methods import (classification_tda_diagram,
    classification_scalar, classification_scalar_svm,
    cvResults2dataframe)
from tqdm import tqdm
from importlib import reload

# import path variables
import config
reload(config)
# don't need the following if reloading; will include it for consistency
from config import data_path_processed, data_path_classification
# import simulation parameters
from config import simulation_parameters

from auxiliary import check_load_file

# %%
# load and set important hyperparameters
processed = simulation_parameters['simulation_processed']
number_run = 0

train_size = simulation_parameters['train_size']
n_cross_val = simulation_parameters['n_cross_val']
random_state_cv = simulation_parameters['random_state_cv']
random_state_classifier = simulation_parameters['random_state_classifier']
                                    
verbose = False

# %%
#
def shorter_names(pd_series):
    return ([x.split('(')[0] for x in pd_series])

# %%
# read in files

filename_base = data_path_classification

file_appendix = simulation_parameters["calculation_method"]
file_appendix_run = f'{simulation_parameters["calculation_method"]}_run-{number_run}_CV-{n_cross_val}'
file_appendix_run += f'_RS-{random_state_cv}-TS-{int(train_size*100)}'

fn = f'diagrams_{file_appendix}.npy'
diags = check_load_file(data_path_processed / fn)  

fn = f'numSimulation_parameters_{file_appendix}.csv'
df_params = check_load_file(data_path_processed / fn) 

le = LabelEncoder()
df_params.loc[:, 'coverage'] = le.fit_transform(df_params.loc[:, 'coverage'].values)

# %%
# create feature lists for the analysis

order_columns = ['feature', 'accuracy', 'f1_macro', 'test',
                'split', 'dim', 'processed',
                'param_Clf', 'param_TDA', 'param_Scaler',
                'param_Dim_reduction', 'params']

# Set all the column names which are NOT parameters
cols_id = ['processed', 'id', 
           'sequence', 'impacts', 
           'coverage', 
           'coverage_old',
           'coverage_normalized', 
           'coverage_labels']

# get the column names for the parameters which will be used 
# in a multi feature classification
cols_conventional = ['Sa', 'Sq', 'Ssk', 'Sku', 'Sz', 'Sdq', 'Sdr']
cols_conventional_select = ['Ssk', 'Sdr']
cols_tda_select= ['entropy_gudhi_0',
                'amplitude_betti_0',
                'max_pers_0',
                'amplitude_landscape_0',
                'amplitude_silhouette_0']
columns_selection = [cols_conventional, cols_conventional_select,
                    cols_tda_select]
# add the single features
columns_selection.extend([['Sa'], ['Sq'], ['Ssk'], ['Sku'], ['Sz'], ['Sdq'], ['Sdr'],
                        ['entropy_gudhi_0'],
                        ['amplitude_betti_0'],
                        ['max_pers_0'],
                        ['amplitude_landscape_0'],
                        ['amplitude_silhouette_0']])

# get the names for the features and classes
cols_names = ['Conv_all', 'Conv_selec', 'tda_selec']
cols_names.extend(['Sa', 'Sq', 'Ssk', 'Sku','Sz', 'Sdq', 'Sdr',
                'entropy_gudhi_0',
                'amplitude_betti_0',
                'max_pers_0',
                'amplitude_landscape_0',
                'amplitude_silhouette_0'])

# %%
# Calculate tda diagram

print('Calculating Classifiers for TDA-representations')

fn = 'tda_representations'
filepath = filename_base / f'interim_{fn}_{file_appendix_run}.csv'
if not (filepath.exists()):

    y = df_params.loc[df_params['processed'] == processed, 'coverage'].values
    groups = df_params.loc[df_params['processed']== processed, 'sequence'].values

    dim = 0 # could also consider dimension 1
    diags_dim = [tmp_diag[tmp_diag[:, 2] == dim, :2]
                for tmp_diag in diags]

    df_tmp, param_cols =\
        classification_tda_diagram(diags_dim, y, groups,
                                train_size=train_size,
                                random_state_cv=random_state_cv,
                                random_state_classifier=random_state_classifier,
                                only_best=True,
                                n_cross_val=n_cross_val,
                                verbose=verbose)
    df_tmp = cvResults2dataframe(df_tmp, param_cols, processed)
    df_tmp['dim'] = dim
    df_tmp['feature'] = 'PersDiag_repre'
    df_tmp['param_Scaler'] = np.nan
    df_tmp['param_Dim_reduction'] = np.nan

    df_result = df_tmp[order_columns]
    df_result.to_csv(filepath, index=False)

# %%
# different classifiers for single/several scalar-valued parameters

print('Calculating Classifiers for scalar-classifiers')

fn = 'scalar'
filepath = filename_base / f'interim_{fn}_{file_appendix_run}.csv'
if not (filepath.exists()):
    count = 0

    for cols_i, cols_classifier in enumerate(tqdm(columns_selection)):
        dim_reduction = False

        X = df_params.loc[df_params['processed']== processed, cols_classifier].values
        y = df_params.loc[df_params['processed']== processed, 'coverage'].values
        groups = df_params.loc[df_params['processed']== processed, 'sequence'].values

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        else:
            if X.shape[1] >= 4:
                dim_reduction = True

        df_tmp, param_cols =\
            classification_scalar(X, y, groups,
                                train_size=train_size,
                                random_state_cv=random_state_cv,
                                random_state_classifier=random_state_classifier,
                                only_best=True,
                                n_cross_val=n_cross_val,
                                dim_reduction=dim_reduction,
                                verbose=verbose)
        df_tmp = cvResults2dataframe(df_tmp, param_cols, processed)

        # get the names of the variables used; tda ones end with 0 or 1
        # Furthermore they are the only ones with "_" inside
        tmp_dim = [int(col.split('_')[-1]) if col.split('_')[-1] in [0, 1]
                else np.nan
                for col in cols_classifier]
        if np.isnan(tmp_dim).any():
            df_tmp['dim'] = np.nan
        elif np.unique(tmp_dim).size > 1:
            df_tmp['dim'] = -1
        else:
            df_tmp['dim'] = np.unique(tmp_dim)[0]

        df_tmp['param_TDA'] = np.nan
        df_tmp['feature'] = cols_names[cols_i]

        # append to Dataframe or create it
        if count == 0:
            df_result = df_tmp.copy(deep=True)
        else:
            df_result = df_result.append(df_tmp)

        count += 1

    df_result = df_result[order_columns]
    df_result.to_csv(filepath, index=False)


# %%
# use scalar svm

print('Calculating Classifiers for scalar-SVM-classifiers')

fn = 'scalar_svm'
filepath = filename_base / f'interim_{fn}_{file_appendix_run}.csv'
if not (filepath.exists()):
    count = 0

    for cols_i, cols_classifier in enumerate(tqdm(columns_selection)):
        dim_reduction = False

        X = df_params.loc[df_params['processed']== processed, cols_classifier].values
        y = df_params.loc[df_params['processed']== processed, 'coverage'].values
        groups = df_params.loc[df_params['processed']== processed, 'sequence'].values

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        else:
            if X.shape[1] >= 4:
                dim_reduction = True

        df_tmp, param_cols =\
            classification_scalar_svm(X, y, groups,
                                    train_size=train_size,
                                    random_state_cv=random_state_cv,
                                    random_state_classifier=random_state_classifier,
                                    only_best=True,
                                    n_cross_val=n_cross_val,
                                    dim_reduction=dim_reduction,
                                    verbose=verbose)
        df_tmp = cvResults2dataframe(df_tmp, param_cols, processed)

        # get the names of the variables used; tda ones end with 0 or 1
        # Furthermore they are the only ones with "_" inside
        tmp_dim = [int(col.split('_')[-1]) if col.split('_')[-1] in [0, 1]
                else np.nan
                for col in cols_classifier]
        if np.isnan(tmp_dim).any():
            df_tmp['dim'] = np.nan
        elif np.unique(tmp_dim).size > 1:
            df_tmp['dim'] = -1
        else:
            df_tmp['dim'] = np.unique(tmp_dim)[0]

        df_tmp['param_TDA'] = np.nan
        df_tmp['feature'] = cols_names[cols_i]

        # append to Dataframe or create it
        if count == 0:
            df_result = df_tmp.copy(deep=True)
        else:
            df_result = df_result.append(df_tmp)

        count += 1

    df_result = df_result[order_columns]
    df_result.to_csv(filepath, index=False)

# %%
# Just getting feedback
print('Fine: program ran.')

## %%
# Read all csv-files. If not all exist, do not save dataframe

type_names = ['tda_representations', 'scalar', 'scalar_svm']

df_all = None
all_files_exist = True
for fn in type_names:
    filepath = filename_base / f'interim_{fn}_{file_appendix_run}.csv'
    
    if not (filepath.exists()):
        print(f'File {filepath} does not exists yet!')
        all_files_exist = False
        break

    df_result = pd.read_csv(filepath)
    if df_all is None:
        df_all = df_result.copy(deep=True)
        print(f'Dataframe created: {fn}')
    else:
        if (df_result.columns == df_all.columns).all():
            df_all = df_all.append(df_result)
            print(f'Appended: {fn}')
        else:
            print(f'File {filepath} '
                + 'has other columns then the rest (or not sorted?)!')
            all_files_exist = False
            break

if all_files_exist:
    df_all = df_all[order_columns]
    
    filepath = filename_base / f'classifier_{file_appendix_run}.csv'
    
    df_all.to_csv(filepath, index=False)
    print('File saved!')

    for fn in type_names:
        filepath = filename_base / f'interim_{fn}_{file_appendix_run}.csv'
        if filepath.exists():
            Path.unlink(filepath)

# %%
