import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler


import joblib as jl

#extraire features dans une dataframe et la visualiser
df= pd.read_csv('alt_acsincome_ca_features_85(1).csv')
lbl = pd.read_csv('alt_acsincome_ca_labels_85.csv')

# Standardisation des attributs numériques: centrer et réduire les données (moyenne = 0, écart-type = 1)pour les algorithmes sensibles à l'échelle des données comme les SVM ou KNN.
scalerX=StandardScaler()
scaler2=scalerX.fit(df)
features_scaled=scaler2.transform(df)
#enregistrement des valeurs du scalerX
jl.dump(scalerX,"scaler.jl")


#étudier la distribution des features
featuresEtudiees = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']
df.describe()
for colonne in featuresEtudiees :
    plt.hist(df[colonne], bins=50, alpha=0.5, label=colonne)
    plt.xlabel('features')
    plt.ylabel('valeurs')
    plt.title('distribution des features')
    plt.legend()
    plt.show()
    plt.savefig('hist'+colonne, dpi=300)

# Préparation des données pour le partitionnement
X = features_scaled
y = lbl.values.ravel()  # Convertir en format numpy (1D) compatible avec StratifiedShuffleSplit

# Partitionnement du jeu de données
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Taille train:", X_train.shape, "Taille test:", X_test.shape)

# Vérification des dimensions après traitement
print(f"\nDimensions des données d'entrée X : {X.shape}")
print(f"Dimensions des labels y : {y.shape}")


# 1) on commence avec le modèle type RandomForest :
rf_model = RandomForestClassifier()
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5).mean()
print("Moyenne des scores de Validation croisée = ", cv_scores_rf)

#on récupère les indices de qualité suivants pour chaque modèle : accuracy, classification_report, confusion_matrix
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)

#acuracy
ac_rf = accuracy_score(y_test, rf_prediction)
#classification
class_rf = classification_report(y_test, rf_prediction)
#matrice de confusion
mat_conf_rf = confusion_matrix(y_test, rf_prediction)

print("Acuracy = ", ac_rf)
print("classification report = ", class_rf)
print("matrice de confusion = ", mat_conf_rf)

#recherche de la meilleure qualité d'apprentissage avec la détermination des hyperparamètres (partie 3.2)
param_rf = {'n_estimators' : [50,100,120], 'criterion':['gini','entropy','log_loss'], 'max_depth':[None, 10, 20], 'min_samples_split':[0.1, 1.0, 2]}
gridSearch_rf = GridSearchCV(RandomForestClassifier(), param_rf, cv=5)
gridSearch_rf.fit(X_train, y_train)
bestParam_rf = gridSearch_rf.best_params_
print("Les meilleurs paramètres sont : ", bestParam_rf)

#enregistrement du meilleur modèle obtenu
jl.dump(gridSearch_rf.best_estimator_ ,'gridSearch_rf.joblib')



# 2) on continue avec le modèle type AdaBoost :
ab_model = AdaBoostClassifier()
cv_scores_ab = cross_val_score(ab_model, X_train, y_train, cv=5).mean()

#on récupère les indices de qualité suivants pour chaque modèle : accuracy, classification_report, confusion_matrix
ab_model.fit(X_train, y_train)
ab_prediction = ab_model.predict(X_test)
#acuracy
ac_ab = accuracy_score(y_test, ab_prediction)
#classification
class_ab = classification_report(y_test, ab_prediction)
#matrice de confusion
mat_conf_ab = confusion_matrix(y_test, ab_prediction)

print("Moyenne des scores de Validation croisée AdaBoost = ", cv_scores_ab)
print("Acuracy AdaBoost = ", ac_ab)
print("classification report AdaBoost = ", class_ab)
print("matrice de confusion AdaBoost = ", mat_conf_ab)

#recherche de la meilleure qualité d'apprentissage avec la détermination des hyperparamètres (partie 3.2)
param_ab = {'n_estimators' : [10,40,100], 'algorithm':['SAMME', 'SAMME.R'], 'learning_rate':[0.2, 0.5, 1.0]}
gridSearch_ab = GridSearchCV(AdaBoostClassifier(), param_ab, cv=5)
gridSearch_ab.fit(X_train, y_train)
bestParam_ab = gridSearch_ab.best_params_
print("Les meilleurs paramètres sont : ", bestParam_ab)

#enregistrement du meilleur modèle obtenu
jl.dump(gridSearch_ab.best_estimator_ ,'gridSearch_ab.joblib')


# 2) on continue avec le modèle type GradientBoosting :

gb_model = GradientBoostingClassifier()
cv_scores_gb = cross_val_score(gb_model, X_train, y_train, cv=5).mean()
print("Moyenne des scores de Validation croisée Gradient Boosting = ", cv_scores_gb)

#on récupère les indices de qualité suivants pour chaque modèle : accuracy, classification_report, confusion_matrix
gb_model.fit(X_train, y_train)
gb_prediction = gb_model.predict(X_test)
#acuracy
ac_gb = accuracy_score(y_test, gb_prediction)
#classification
class_gb = classification_report(y_test, gb_prediction)
#matrice de confusion
mat_conf_gb = confusion_matrix(y_test, gb_prediction)

print("Acuracy Gradient Boosting = ", ac_gb)
print("classification report Gradient Boosting = ", class_gb)
print("matrice de confusion Gradient Boosting = ", mat_conf_gb)

#recherche de la meilleure qualité d'apprentissage avec la détermination des hyperparamètres (partie 3.2)
param_gb = {'n_estimators' : [20,50,100], 'loss':['log_loss', 'exponential'], 'learning_rate':[0.2, 0.5, 1.0], 'criterion':['friedman_mse', 'squarred_error'], 'subsample':[0.5, 1.0, 1.5]}
gridSearch_gb = GridSearchCV(GradientBoostingClassifier(), param_gb, cv=5)
gridSearch_gb.fit(X_train, y_train)
bestParam_gb = gridSearch_gb.best_params_
print("Les meilleurs paramètres sont : ", bestParam_gb)

#enregistrement du meilleur modèle obtenu
jl.dump(gridSearch_gb.best_estimator_ ,'gridSearch_gb.joblib')


