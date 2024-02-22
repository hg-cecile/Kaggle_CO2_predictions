import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro


import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from category_encoders import CatBoostEncoder, LeaveOneOutEncoder, TargetEncoder

import lightgbm as lgb

import preprocessing as prepro
import modele

from typing import List, Set, Tuple

pd.set_option('display.max_columns', None)

# nombre de NaN
def infos_valeurs_manquantes(df_train, df_test):
    print(f"Nombre de NaN pour df_train : {df_train.isna().sum()}")
    print(f"Nombre de NaN pour df_test : {df_test.isna().sum()}")

# taux de valeurs manquantes
def taux_valeurs_manquantes(df) :
    taux = df.isna().sum()/len(df)*100
    print(taux)
    return(taux)

# récupérer les variables ayant au moins 50% de NaN
def variables_50pct_NaN(df, taux_NaN):
    cols_to_drop = [col for col in df.columns if taux_NaN[col] > 50]
    return cols_to_drop

# tracer la distribution des variables souhaitées
def graphs_distributions(variables,df):

    num_rows = 3
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 12))

    for i in range(num_rows):
        for j in range(num_cols):
            if variables:
                variable_actuelle = variables.pop(0)
                variable_a_tracer = df[variable_actuelle]
        
                # Filtrer les valeurs NaN
                variable_sans_nan = variable_a_tracer[~np.isnan(variable_a_tracer)]
            
                axs[i, j].hist(variable_sans_nan, bins='auto', alpha=0.7, color='blue', edgecolor='black')
                axs[i, j].set_title(f'Distribution de {variable_actuelle}')
                axs[i, j].set_xlabel('Valeurs')
                axs[i, j].set_ylabel('Fréquence')
            else:
                axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()


# LabelEncoder
def encodage(df_train, df_test, col_categoricals_train, col_categoricals_test):

    # 1ère partie : on traite les variables qui ont des modalités communes dans df_train et df_test, car le LabelEncoder ne peut pas traiter quand la modalité est nouvelle dans df_test
    liste_variables_modalites_communes = []
    
    # pour chaque variable de df_test, on regarde les modalités qui sont différentes de df_train
    for col in col_categoricals_test:
        liste_modalite_test=list(df_test[col].unique())
        liste_modalite_train=list(df_train[col].unique())
        # si ça print une liste vide alors toutes les modalités de df_test sont dans df_train
        liste_var_moda_diff = [col for col in liste_modalite_test if col not in liste_modalite_train]
        # si la différence est nulle, alors toutes les modalités sont communes et on ajoute la variable à la liste
        if len(liste_var_moda_diff) == 0 :
            liste_variables_modalites_communes.append(col)
        print(f"Variable : {col}, longueur : {len(liste_var_moda_diff)}, modalités différentes : {liste_var_moda_diff}")

    
    liste_variables_modalites_non_communes = [col for col in col_categoricals_train if col not in liste_variables_modalites_communes]
    
    encoder=LabelEncoder()

    for col in liste_variables_modalites_communes:
        df_train[col] = encoder.fit_transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])

    # 2ème partie : on traite les variables dont les modalités sont inconnues dans df_test
    label_encoder = LabelEncoder()
    for col in liste_variables_modalites_non_communes :
    
        # on met dans un dataframe temporaire df_train[col] et df_test[col]
        df_concat = np.concatenate([df_train[col], df_test[col]])

        # puisque tout est rassemblé dans le même dataframe, on fit_transform pour éviter les modalités inconnues
        fit_transform_df_concat = label_encoder.fit_transform(df_concat)

        # on écrase la valeur de df_train[col] et df_test[col] par la colonne encodée, en récupérant la bonne longueur pour chaque df respectif
        df_train[col] = fit_transform_df_concat[:len(df_train[col])]
        df_test[col] = fit_transform_df_concat[len(df_train[col]):]

    return df_train, df_test
