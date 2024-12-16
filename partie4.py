import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_selection import chi2
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import joblib as jl

df= pd.read_csv('alt_acsincome_ca_features_85(1).csv')
lbl = pd.read_csv('alt_acsincome_ca_labels_85.csv')

scalerX=StandardScaler()
scaler2=scalerX.fit(df)
features_scaled=scaler2.transform(df)

X = features_scaled
y = lbl.values.ravel()  

# Partitionnement du jeu de données
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Taille train:", X_train.shape, "Taille test:", X_test.shape)

# Fusion des données d'entraînement complètes (X_train_all et y_train_all)
X_train_df = pd.DataFrame(X_train, columns=df.columns)  # Keep original feature names
X_train_df["PINCP"] = y_train  # Add the target column

# Calcul de la matrice de corrélation pour les données d'entraînement complètes
correlation_matrix = X_train_df.corr()

# Affichage de la matrice de corrélation dans la console
print("Matrice de corrélation :")
print(correlation_matrix)

# Visualisation de la matrice de corrélation pour les données initiales
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de corrélation des données initiales')
plt.show()

#I. RandomForest############################################################################################################""
rf = jl.load('gridSearch_rf.joblib')

rf.fit(X_train, y_train)  # Entraînement du modèle avec les meilleurs paramètres 

# 2. Prédiction sur l'ensemble de test complet (X_test_all)
y_pred_rf = rf.predict(X_test)
print("Les prédictions : ", y_pred_rf)

# 3. Création d'une DataFrame pour les prédictions
df_y_pred_rf = pd.DataFrame(y_pred_rf, columns=['PINCP'])

# 4. Fusion des prédictions avec les features de test
# Convert X_test (NumPy array) into a DataFrame with the original column names
X_test_df = pd.DataFrame(X_test, columns=df.columns)

# Merge X_test_df (DataFrame) with df_y_pred_rf
merged_rf = pd.concat([X_test_df, df_y_pred_rf], axis=1)


# 5. Calcul de la matrice de corrélationn avec les données produites par les modèles d’apprentissage
correlation_matrix_rf = merged_rf.corr()

# 6. Affichage de la matrice de corrélation dans la console
print("Matrice de corrélation après entraînement avec RandomForest :")
print(correlation_matrix_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_rf, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de corrélation après entraînement avec RandomForest')
plt.show()

#classement des différents attributs du jeu de données par ordre d’importance
importance_rf = permutation_importance(rf, X_test, df_y_pred_rf, random_state=42).importances_mean

for i in range(len(importance_rf)):
    print(f"Feature {i}: ", importance_rf[i])

features = ['Feature 0:AGEP', 'Feature 1:COW', 'Feature 2:SCHL', 'Feature 3:MAR', 'Feature 4:OCCP',
            'Feature 5: POBP', 'Feature 6:RELP', 'Feature 7:WKHP', 'Feature 8:SEX', 'Feature 9:RAC1P']
plt.figure(figsize=(10, 6))
plt.bar(features, importance_rf, color='skyblue')
plt.title('Feature Importance Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right') 
plt.show()

#II. AdaBoost############################################################################################################""
ab = jl.load('gridSearch_ab.joblib')
ab.fit(X_train, y_train)  # Entraînement du modèle

# 2. Prédiction sur l'ensemble de test complet (X_test_all)
y_pred_ab = ab.predict(X_test)

# 3. Création d'une DataFrame pour les prédictions
df_y_pred_ab = pd.DataFrame(y_pred_ab, columns=['PINCP'])

# 4. Fusion des prédictions avec les features de test
merged_ab = pd.concat([X_test_df, df_y_pred_ab], axis=1)

# 5. Calcul de la matrice de corrélation
correlation_matrix_ab = merged_ab.corr()

# 6. Affichage de la matrice de corrélation 
print("Matrice de corrélation après entraînement avec AdaBoost :")
print(correlation_matrix_ab)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_ab, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de corrélation après entraînement avec AdaBoost')
plt.show()

#classement des différents attributs du jeu de données par ordre d’importance
importance_ab = permutation_importance(ab, X_test, df_y_pred_ab, random_state=42).importances_mean

for i in range(len(importance_ab)):
    print(f"Feature {i}: ", importance_ab[i])

features = ['Feature 0:AGEP', 'Feature 1:COW', 'Feature 2:SCHL', 'Feature 3:MAR', 'Feature 4:OCCP',
            'Feature 5: POBP', 'Feature 6:RELP', 'Feature 7:WKHP', 'Feature 8:SEX', 'Feature 9:RAC1P']
plt.figure(figsize=(10, 6))
plt.bar(features, importance_ab, color='red')
plt.title('Feature Importance AdaBoost')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right') 
plt.show()

#III. GradientBoosting############################################################################################################""
gb = jl.load('gridSearch_gb.joblib')

gb.fit(X_train, y_train)  # Entraînement du modèle

# 2. Prédiction sur l'ensemble de test complet (X_test_all)
y_pred_gb = gb.predict(X_test)

# 3. Création d'une DataFrame pour les prédictions
df_y_pred_gb = pd.DataFrame(y_pred_gb, columns=['PINCP'])

# 4. Fusion des prédictions avec les features de test
merged_gb = pd.concat([X_test_df, df_y_pred_gb], axis=1)

# 5. Calcul de la matrice de corrélation
correlation_matrix_gb = merged_gb.corr()

# 6. Affichage de la matrice de corrélation dans la console
print("Matrice de corrélation après entraînement avec GB :")
print(correlation_matrix_gb)

# 7. Visualisation de la matrice de corrélation avec un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_gb, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de corrélation après entraînement avec GB')
plt.show()

#classement des différents attributs du jeu de données par ordre d’importance
importance_gb = permutation_importance(gb, X_test, df_y_pred_gb, random_state=42).importances_mean
for i in range(len(importance_gb)):
    print(f"Feature {i}: ", importance_gb[i])

features = ['Feature 0:AGEP', 'Feature 1:COW', 'Feature 2:SCHL', 'Feature 3:MAR', 'Feature 4:OCCP',
            'Feature 5: POBP', 'Feature 6:RELP', 'Feature 7:WKHP', 'Feature 8:SEX', 'Feature 9:RAC1P']
plt.figure(figsize=(10, 6))
plt.bar(features, importance_gb, color='green')
plt.title('Feature Importance GB')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right') 
plt.show()

