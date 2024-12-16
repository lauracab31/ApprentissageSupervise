from matplotlib import pyplot as plt
import pandas as pd
import joblib as jl
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

# Chargement des données
df = pd.read_csv('alt_acsincome_ca_features_85(1).csv')
lbl = pd.read_csv('alt_acsincome_ca_labels_85.csv')

# Standardisation des attributs numériques
scalerX = StandardScaler()
features_scaled = scalerX.fit_transform(df)
jl.dump(scalerX, "scaler.jl")

# Préparation des données pour le partitionnement
X = features_scaled
y = lbl.values.ravel()  # Convertir en format numpy compatible avec StratifiedShuffleSplit

# Partitionnement du jeu de données
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Création de DataFrames pour les features avec les indices des splits
X_train_df = pd.DataFrame(X_train, columns=df.columns)
X_test_df = pd.DataFrame(X_test, columns=df.columns)

y_train_df = pd.DataFrame(y_train, columns=['PINCP'])
y_test_df = pd.DataFrame(y_test, columns=['PINCP'])

# Ajout de la colonne SEX en fonction des indices
X_train_df['SEX'] = df['SEX'].iloc[train_index].values
X_test_df['SEX'] = df['SEX'].iloc[test_index].values

# Ajout de la colonne PINCP pour simplifier l'analyse
X_train_df['PINCP'] = y_train_df['PINCP'].values
X_test_df['PINCP'] = y_test_df['PINCP'].values

# Séparation des données par sexe (SEX == 1 pour hommes, SEX == 2 pour femmes)
X_train_homme = X_train_df[X_train_df['SEX'] == 1]
X_train_femme = X_train_df[X_train_df['SEX'] == 2]

X_test_homme = X_test_df[X_test_df['SEX'] == 1]
X_test_femme = X_test_df[X_test_df['SEX'] == 2]

# Séparation des features (X) et de la cible (y) 
y_train_homme = X_train_homme['PINCP']
X_train_homme = X_train_homme.drop(columns=['PINCP'])

y_train_femme = X_train_femme['PINCP']
X_train_femme = X_train_femme.drop(columns=['PINCP'])

y_test_homme = X_test_homme['PINCP']
X_test_homme = X_test_homme.drop(columns=['PINCP'])

y_test_femme = X_test_femme['PINCP']
X_test_femme = X_test_femme.drop(columns=['PINCP'])

# Chargement des modèles pré-entraînés
rf = jl.load('gridSearch_rf.joblib')
ab = jl.load('gridSearch_ab.joblib')
gb = jl.load('gridSearch_gb.joblib')

# Entraînement et évaluation des modèles avec la colonne SEX (hommes)
rf.fit(X_train_homme, y_train_homme)
ab.fit(X_train_homme, y_train_homme)
gb.fit(X_train_homme, y_train_homme)

y_pred_rf_homme = rf.predict(X_test_homme)
y_pred_ab_homme = ab.predict(X_test_homme)
y_pred_gb_homme = gb.predict(X_test_homme)

print("Matrice de confusion Random Forest pour les hommes:\n", confusion_matrix(y_test_homme, y_pred_rf_homme, normalize='all'))
print("Matrice de confusion AdaBoost pour les hommes:\n", confusion_matrix(y_test_homme, y_pred_ab_homme, normalize='all'))
print("Matrice de confusion Gradient Boosting pour les hommes:\n", confusion_matrix(y_test_homme, y_pred_gb_homme, normalize='all'))

# Entraînement et évaluation des modèles avec la colonne SEX (femmes)
rf.fit(X_train_femme, y_train_femme)
ab.fit(X_train_femme, y_train_femme)
gb.fit(X_train_femme, y_train_femme)

y_pred_rf_femme = rf.predict(X_test_femme)
y_pred_ab_femme = ab.predict(X_test_femme)
y_pred_gb_femme = gb.predict(X_test_femme)

print("Matrice de confusion Random Forest pour les femmes:\n", confusion_matrix(y_test_femme, y_pred_rf_femme, normalize='all'))
print("Matrice de confusion AdaBoost pour les femmes:\n", confusion_matrix(y_test_femme, y_pred_ab_femme, normalize='all'))
print("Matrice de confusion Gradient Boosting pour les femmes:\n", confusion_matrix(y_test_femme, y_pred_gb_femme, normalize='all'))

# Suppression de la colonne SEX pour tester l'équité des modèles
train_sans_sex_homme = X_train_homme.drop(columns=['SEX'])
train_sans_sex_femme = X_train_femme.drop(columns=['SEX'])

test_sans_sex_homme = X_test_homme.drop(columns=['SEX'])
test_sans_sex_femme = X_test_femme.drop(columns=['SEX'])

# Entraînement sans la colonne SEX
rf.fit(train_sans_sex_homme, y_train_homme)
y_pred_rf_sansSex_homme = rf.predict(test_sans_sex_homme)
print("Matrice de confusion Random Forest sans SEX pour les hommes:\n", confusion_matrix(y_test_homme, y_pred_rf_sansSex_homme, normalize='all'))

rf.fit(train_sans_sex_femme, y_train_femme)
y_pred_rf_sansSex_femme = rf.predict(test_sans_sex_femme)
print("Matrice de confusion Random Forest sans SEX pour les femmes:\n", confusion_matrix(y_test_femme, y_pred_rf_sansSex_femme, normalize='all'))

ab.fit(train_sans_sex_homme, y_train_homme)
y_pred_ab_sansSex_homme = ab.predict(test_sans_sex_homme)
print("Matrice de confusion AdaBoost sans SEX pour les hommes:\n", confusion_matrix(y_test_homme, y_pred_ab_sansSex_homme, normalize='all'))

ab.fit(train_sans_sex_femme, y_train_femme)
y_pred_ab_sansSex_femme = ab.predict(test_sans_sex_femme)
print("Matrice de confusion AdaBoost sans SEX pour les femmes:\n", confusion_matrix(y_test_femme, y_pred_ab_sansSex_femme, normalize='all'))

gb.fit(train_sans_sex_homme, y_train_homme)
y_pred_gb_sansSex_homme = gb.predict(test_sans_sex_homme)
print("Matrice de confusion Gradient Boosting sans SEX pour les hommes:\n", confusion_matrix(y_test_homme, y_pred_gb_sansSex_homme, normalize='all'))

gb.fit(train_sans_sex_femme, y_train_femme)
y_pred_gb_sansSex_femme = gb.predict(test_sans_sex_femme)
print("Matrice de confusion Gradient Boosting sans SEX pour les femmes:\n", confusion_matrix(y_test_femme, y_pred_gb_sansSex_femme, normalize='all'))