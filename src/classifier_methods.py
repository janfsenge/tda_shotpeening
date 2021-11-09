
"""
Provide different methods to calculate classifications.

All methods employ scikit Pipelines and Gridsearch.

"""

# %%
from gudhi.representations import (BettiCurve, Silhouette, TopologicalVector,
                                   PersistenceImage, Landscape)

import pandas as pd
import os
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

# %%
#

def cvResults2dataframe(cv_results,
                        param_cols,
                        processed='not_specified'):
    """Pivot and unpivot the dataframe given from GridSearchCv.

    Combine the data to join the different columns of the form
    split_* from the cv_results of GridSearchCv 
    Parameters
    ----------
    cv_results : [type]
        [description]
    param_cols : [type]
        [description]
    processed : str, optional
        [description], by default 'not_specified'

    Returns
    -------
    [type]
        [description]
    """
    df_scorer = pd.DataFrame(cv_results)
    split_cols = [col for col in df_scorer.columns
                  if 'split' in col]

    df_tmp = df_scorer.melt(id_vars=param_cols,
                            value_vars=split_cols)
    # get two new columns indicating test/train and accuracy/f1-macro
    df_tmp['test'] = [elt.split('_')[1] for elt in df_tmp['variable']]
    df_tmp['scorer'] = ['_'.join(elt.split('_')[2:])
                        for elt in df_tmp['variable']]
    df_tmp['split'] = [elt.split('_')[0].split('split')[1]
                       for elt in df_tmp['variable']]

    index_cols = param_cols + ['split', 'test', 'scorer']
    df_tmp = df_tmp[index_cols + ['variable', 'value']]
    df_tmp = df_tmp.set_index(index_cols)['value'].unstack().reset_index()
    df_tmp['processed'] = processed

    return (df_tmp)

# %%
# classifier on diagrams

def classification_tda_diagram(diags, y, groups,
                               train_size=0.5,
                               random_state_cv = None,
                               random_state_classifier = None,
                               only_best=True,
                               n_cross_val=5,
                               verbose=False):

    # Classifiaction using other tda representations
    pipe = Pipeline([  # ('Standardization', StandardScaler()),
        ('TDA', BettiCurve()),
        ("Clf", SVC())])

    # Parameters of pipeline. This is the place where you specify the methods you want to use to handle diagrams
    param = [
        {
            "TDA":                 [BettiCurve()],
            "Clf":           [SVC(), RandomForestClassifier()],
            "Clf__random_state": [random_state_classifier]
        },

        {
            "TDA":                 [Landscape()],
            "TDA__resolution":     [100, 200],
            "Clf":           [SVC(), RandomForestClassifier()],
            "Clf__random_state": [random_state_classifier]
        },

        {
            "TDA":                 [Silhouette()],
            "TDA__resolution":     [100, 200],
            "Clf":           [SVC(), RandomForestClassifier()],
            "Clf__random_state": [random_state_classifier]
        }
    ]

    scoring = ['accuracy', 'f1_macro']
    gss = GroupShuffleSplit(n_splits=n_cross_val,
                            train_size=train_size,
                            random_state=random_state_cv)
    clf_repre = GridSearchCV(pipe,
                             param_grid=param,
                             cv=gss.split(diags, y, groups),
                             scoring=scoring,
                             refit='accuracy',
                             return_train_score=True,
                             n_jobs=-1,
                             verbose=verbose)

    clf_repre.fit(diags, y)

    result = pd.DataFrame(clf_repre.cv_results_)
    result['param_Clf'] = result['param_Clf'].apply(str)
    result['param_TDA'] = result['param_TDA'].apply(str)
    result['params'] = result['params'].apply(str)

    if only_best:
        idx_best = []
        # For each different classifier find the best version of it
        for paramTDA in np.unique(result['param_TDA']):
            df_tmp = result[result['param_TDA'] == paramTDA]
            df_tmp = df_tmp.sort_values(['mean_test_accuracy',
                                         'mean_test_f1_macro',
                                         'std_test_accuracy',
                                         'std_test_f1_macro',
                                         'mean_train_accuracy',
                                         'mean_train_f1_macro'],
                                        ascending=False)
            idx_best.append(df_tmp.index[0])
        result = result.loc[idx_best]
    result = result.reset_index(drop=True)

    return (result, ['param_Clf', 'param_TDA', 'params'])


# %%
#

def classification_scalar(X, y, groups,
                          train_size=0.5,
                          random_state_cv = None,
                          random_state_classifier = None,
                          only_best=True,
                          n_cross_val=5,
                          dim_reduction=False,
                          verbose=False):
    # Parameters of pipeline. This is the place where you specify the methods you want to use to handle diagrams
    classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(random_state=random_state_classifier),  # max_depth=5),
        RandomForestClassifier(random_state=random_state_classifier),  # max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=2000, random_state=random_state_classifier),
        # GaussianNB(),
    ]
    if dim_reduction:
        pipe = Pipeline([
            ('Scaler', StandardScaler()),
            ('Dim_reduction', None),
            ('Clf', SVC())])
        param = [
            {'Scaler': [StandardScaler(),
                        MinMaxScaler()],
             'Dim_reduction': [None,
                               LinearDiscriminantAnalysis(n_components=2),
                               PCA()],
             'Clf': classifiers
             }
        ]
    else:
        pipe = Pipeline([
            ('Scaler', StandardScaler()),
            ('Dim_reduction', None),
            ('Clf', SVC())])
        param = [
            {'Scaler': [StandardScaler(),
                        MinMaxScaler()],
             'Dim_reduction': [None],
             'Clf': classifiers
             }
        ]
    scoring = ['accuracy', 'f1_macro']
    gss = GroupShuffleSplit(n_splits=n_cross_val,
                            train_size=train_size,
                            random_state=random_state_cv)
    clf_scalar = GridSearchCV(pipe,
                              param_grid=param,
                              cv=gss.split(X, y, groups),
                              scoring=scoring,
                              refit='accuracy',
                              return_train_score=True,
                              n_jobs=-1,
                              verbose=verbose)

    clf_scalar.fit(X, y)

    result = pd.DataFrame(clf_scalar.cv_results_)
    result['param_Scaler'] = result['param_Scaler'].apply(str)
    result['param_Dim_reduction'] = result['param_Dim_reduction'].apply(str)
    result['param_Clf'] = result['param_Clf'].apply(str)
    result['params'] = result['params'].apply(str)

    if only_best:
        idx_best = []
        # For each different classifier find the best version of it
        for paramClf in np.unique(result['param_Clf']):
            df_tmp = result[result['param_Clf'] == paramClf]
            df_tmp = df_tmp.sort_values(['mean_test_accuracy',
                                         'mean_test_f1_macro',
                                         'std_test_accuracy',
                                         'std_test_f1_macro',
                                         'mean_train_accuracy',
                                         'mean_train_f1_macro'],
                                        ascending=False)
            idx_best.append(df_tmp.index[0])
        result = result.loc[idx_best]
    result = result.reset_index(drop=True)

    return (result, ['param_Scaler', 'param_Dim_reduction',
                     'param_Clf', 'params'])

# %%
# Try several support vector classifiers with the aim to get the best hyperparameters

def classification_scalar_svm(X, y, groups,
                              train_size=0.5,
                              random_state_cv = None,
                              random_state_classifier = None,
                              only_best=True,
                              n_cross_val=5,
                              dim_reduction=False,
                              verbose=False):

    # Parameters of pipeline.
    classifiers = [SVC(kernel='linear', C=0.1),
                   SVC(kernel='linear', C=1),
                   SVC(kernel='linear', C=10),
                   #
                   SVC(kernel='rbf', C=0.1, gamma=0.1),
                   SVC(kernel='rbf', C=1, gamma=0.1),
                   SVC(kernel='rbf', C=10, gamma=0.1),
                   SVC(kernel='rbf', C=0.1, gamma='scale'),
                   SVC(kernel='rbf', C=1, gamma='scale'),
                   SVC(kernel='rbf', C=10, gamma='scale'),
                   SVC(kernel='rbf', C=0.1, gamma='auto'),
                   SVC(kernel='rbf', C=1, gamma='auto'),
                   SVC(kernel='rbf', C=10, gamma='auto'),
                   #
                   SVC(kernel='poly', C=0.1, gamma=0.1),
                   SVC(kernel='poly', C=1, gamma=0.1),
                   SVC(kernel='poly', C=10, gamma=0.1),
                   SVC(kernel='poly', C=0.1, gamma='scale'),
                   SVC(kernel='poly', C=1, gamma='scale'),
                   SVC(kernel='poly', C=10, gamma='scale'),
                   SVC(kernel='poly', C=0.1, gamma='auto'),
                   SVC(kernel='poly', C=1, gamma='auto'),
                   SVC(kernel='poly', C=10, gamma='auto'),
                   #
                   SVC(kernel='sigmoid', C=0.1, gamma=0.1),
                   SVC(kernel='sigmoid', C=1, gamma=0.1),
                   SVC(kernel='sigmoid', C=10, gamma=0.1),
                   SVC(kernel='sigmoid', C=0.1, gamma='scale'),
                   SVC(kernel='sigmoid', C=1, gamma='scale'),
                   SVC(kernel='sigmoid', C=10, gamma='scale'),
                   SVC(kernel='sigmoid', C=0.1, gamma='auto'),
                   SVC(kernel='sigmoid', C=1, gamma='auto'),
                   SVC(kernel='sigmoid', C=10, gamma='auto')
                   ]

    if dim_reduction:
        pipe = Pipeline([
            ('Scaler', StandardScaler()),
            ('Dim_reduction', None),
            ('Clf', SVC())])
        param = [
            {'Scaler': [StandardScaler(),
                        MinMaxScaler()],
             'Dim_reduction': [None,
                               LinearDiscriminantAnalysis(n_components=2),
                               PCA()],
             'Clf': classifiers,
             'Clf__random_state': [random_state_classifier]
             }
        ]
    else:
        pipe = Pipeline([
            ('Scaler', StandardScaler()),
            ('Dim_reduction', None),
            ('Clf', SVC())])
        param = [
            {'Scaler': [StandardScaler(),
                        MinMaxScaler()],
             'Dim_reduction': [None],
             'Clf': classifiers,
             'Clf__random_state': [random_state_classifier]
             }
        ]

    scoring = ['accuracy', 'f1_macro']
    gss = GroupShuffleSplit(n_splits=n_cross_val,
                            train_size=train_size,
                            random_state=random_state_cv)
    clf_scalar = GridSearchCV(pipe,
                              param_grid=param,
                              cv=gss.split(X, y, groups),
                              scoring=scoring,
                              refit='accuracy',
                              return_train_score=True,
                              n_jobs=-1,
                              verbose=verbose)
    clf_scalar.fit(X, y)

    result = pd.DataFrame(clf_scalar.cv_results_)
    result['param_Scaler'] = result['param_Scaler'].apply(str)
    result['param_Dim_reduction'] = result['param_Dim_reduction'].apply(str)
    result['param_Clf'] = result['param_Clf'].apply(str)
    result['params'] = result['params'].apply(str)

    if only_best:
        idx_best = []
        # For each different classifier find the best version of it
        for paramClf in np.unique(result['param_Clf']):
            df_tmp = result[result['param_Clf'] == paramClf]
            df_tmp = df_tmp.sort_values(['mean_test_accuracy',
                                         'mean_test_f1_macro',
                                         'std_test_accuracy',
                                         'std_test_f1_macro',
                                         'mean_train_accuracy',
                                         'mean_train_f1_macro'],
                                        ascending=False)
            idx_best.append(df_tmp.index[0])
        result = result.loc[idx_best]
    result = result.reset_index(drop=True)

    return (result, ['param_Scaler', 'param_Dim_reduction',
                     'param_Clf', 'params'])
