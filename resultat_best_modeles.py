
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler


import joblib as jl

#extraire features dans une dataframe et la visualiser
df= pd.read_csv('alt_acsincome_ca_features_85(1).csv')
lbl = pd.read_csv('alt_acsincome_ca_labels_85.csv')

#print (df)

# Standardisation des attributs numériques: centrer et réduire les données (moyenne = 0, écart-type = 1)pour les algorithmes sensibles à l'échelle des données comme les SVM ou KNN.
scalerX=StandardScaler()
scaler2=scalerX.fit(df)
features_scaled=scaler2.transform(df)
#enregistrement des valeurs du scalerX
jl.dump(scalerX,"scaler.jl")

# Préparation des données pour le partitionnement
X = features_scaled
y = lbl.values.ravel()  # Convertir en format numpy (1D) compatible avec StratifiedShuffleSplit

# Partitionnement du jeu de données
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Taille train:", X_train.shape, "Taille test:", X_test.shape)


# 1) on commence avec le modèle type RandomForest :
rf_model = jl.load('gridSearch_rf.joblib')
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5).mean()


#on récupère les indices de qualité suivants pour chaque modèle : accuracy, classification_report, confusion_matrix
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)

#acuracy
ac_rf = accuracy_score(y_test, rf_prediction)
#classification
class_rf = classification_report(y_test, rf_prediction)
#matrice de confusion
mat_conf_rf = confusion_matrix(y_test, rf_prediction)

print("Moyenne des scores de Validation croisée meilleur modèle RandomForest = ", cv_scores_rf)
print("Acuracy meilleur modèle RandomForest= ", ac_rf)
print("classification report meilleur modèle RandomForest= \n", class_rf)
print("matrice de confusion meilleur modèle RandomForest= ", mat_conf_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(mat_conf_rf, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted -50k", "Predicted +50k"],
            yticklabels=["Actual -50k", "Actual +50k"])
# Add labels, title, and formatting
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Matrice de confusion RandomForest avec les meilleurs paramètres")
plt.show()



# 2) on continue avec le modèle type AdaBoost :
ab_model = jl.load('gridSearch_ab.joblib')
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

print("Moyenne des scores de Validation croisée meilleur modèle AdaBoost= ", cv_scores_ab)
print("Acuracy AdaBoost meilleur modèle = ", ac_ab)
print("classification report AdaBoost meilleur modèle = \n", class_ab)
print("matrice de confusion AdaBoost meilleur modèle = ", mat_conf_ab)

plt.figure(figsize=(8, 6))
sns.heatmap(mat_conf_ab, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted -50k", "Predicted +50k"],
            yticklabels=["Actual -50k", "Actual +50k"])

# Add labels, title, and formatting
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Matrice de confusion AdaBoost avec les meilleurs paramètres")
plt.show()



# 2) on continue avec le modèle type GradientBoosting :

gb_model = jl.load('gridSearch_gb.joblib')
cv_scores_gb = cross_val_score(gb_model, X_train, y_train, cv=5).mean()


#on récupère les indices de qualité suivants pour chaque modèle : accuracy, classification_report, confusion_matrix
gb_model.fit(X_train, y_train)
gb_prediction = gb_model.predict(X_test)
#acuracy
ac_gb = accuracy_score(y_test, gb_prediction)
#classification
class_gb = classification_report(y_test, gb_prediction)
#matrice de confusion
mat_conf_gb = confusion_matrix(y_test, gb_prediction)

print("Moyenne des scores de Validation croisée meilleur modèle GradientBoosting= ", cv_scores_gb)
print("Acuracy GradientBoosting meilleur modèle= ", ac_gb)
print("classification report GradientBoosting meilleur modèle = \n", class_gb)
print("matrice de confusion GradientBoosting meilleur modèle = ", mat_conf_gb)

plt.figure(figsize=(8, 6))
sns.heatmap(mat_conf_gb, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted -50k", "Predicted +50k"],
            yticklabels=["Actual -50k", "Actual +50k"])
# Add labels, title, and formatting
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Matrice de confusion GradientBoosting avec les meilleurs paramètres")
plt.show()
