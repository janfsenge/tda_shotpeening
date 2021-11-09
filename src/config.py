# config.py

"""Set the specific paths used in the ipynb modules."""
# %%

from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


project_path = get_project_root()
data_path_raw = project_path / 'data' / 'raw'
data_path_raw_simulation = project_path / \
    'data' / 'raw' / 'armco_numerical_simulation'
data_path_interim = project_path / 'data' / 'interim'
data_path_processed = project_path / 'data' / 'processed'
data_path_classification = project_path / 'data' / 'classification'

figures_path = project_path / 'figures'

train_size = 0.5
n_cross_val = 10
random_state_cv = 42
random_state_classifier = 42


# %%
# SET THE SIMULATION PARAMETERS

# the standard for the paper is 'nom_filt'
calculation_method = 'nom_filt'  

# # parameters to emulate the paper which this is based upon
# if calculation_method == 'consistent':
#     simulation_processed = 'nom_filt'

#     bigger_grid = True
#     interpolation_size_preprocess = 0 # is a bad idea -->
#     interpolation_size_postprocess = 1000
#     cut_off_wavelength = 0.8

# # parameters so that less assumptions are made, more precise.
# if calculation_method == 'filt':
#     # preprocess
#     interpolation_size_preprocess = 1+73*14
#     bigger_grid = True

#     # postprocess
#     simulation_processed = 'filt'
#     interpolation_size_postprocess = 0
#     cut_off_wavelength = 0.8

# %%
# parameters so that less assumptions are made, more precise;
if calculation_method == 'nom_filt':
    # preprocess
    interpolation_size_preprocess = 1+73*14
    bigger_grid = True

    # postprocess
    simulation_processed = 'nom_filt'
    interpolation_size_postprocess = 0
    cut_off_wavelength = 0.8

# # take 0.25 cut off wavelength
# if calculation_method == 'filt_cut':
#     # preprocess
#     interpolation_size_preprocess = 1+73*14
#     bigger_grid = True

#     # postprocess
#     simulation_processed = 'filt'
#     interpolation_size_postprocess = 0
#     cut_off_wavelength = 0.25

# %%
# # take 0.25 cut off wavelength
# if calculation_method == 'nom_filt_cut':
#     # preprocess
#     interpolation_size_preprocess = 1+73*14
#     bigger_grid = True

#     # postprocess
#     simulation_processed = 'nom_filt'
#     interpolation_size_postprocess = 0
#     cut_off_wavelength = 0.25


simulation_parameters = {'simulation_processed': simulation_processed,
                         'bigger_grid': bigger_grid,
                         'interpolation_size_preprocess': interpolation_size_preprocess,
                         'interpolation_size_postprocess': interpolation_size_postprocess,
                         'cut_off_wavelength': cut_off_wavelength,
                         'calculation_method': calculation_method,
                         'train_size': train_size,
                         'n_cross_val': n_cross_val,
                         'random_state_cv': random_state_cv,
                         'random_state_classifier': random_state_classifier,
                         }
