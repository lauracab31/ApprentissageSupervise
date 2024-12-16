from matplotlib import pyplot as plt
import pandas as pd
import joblib as jl
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

df = pd.read_csv('alt_acsincome_ca_features_85(1).csv')
lbl = pd.read_csv('alt_acsincome_ca_labels_85.csv')

data = pd.concat([df, lbl], axis=1)
num_samples = int(len(data) * 0.1)

hommes = data[data['SEX'] == 1]
femmes = data[data['SEX'] == 2]

taux_total = data['PINCP'].mean() * 100
print(f"Taux global d'individus ayant un revenu supérieur à 50 000 dollars: {taux_total:.2f}%")

taux_hommes = hommes['PINCP'].mean() * 100
print(f"Taux d'hommes ayant un revenu supérieur à 50 000 dollars: {taux_hommes:.2f}%")

taux_femmes = femmes['PINCP'].mean() * 100
print(f"Taux de femmes ayant un revenu supérieur à 50 000 dollars: {taux_femmes:.2f}%")


