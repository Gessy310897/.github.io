with open("data_pipeline.py", "w") as f:
    f.write('''\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def catalogage(dataset, question):
    question = question.lower()
    if "colonnes" in question:
        return dataset.columns.tolist()
    elif "taille" in question or "dimensions" in question:
        return dataset.shape
    elif "types de données" in question:
        return dataset.dtypes
    elif "valeurs manquantes" in question:
        return dataset.isnull().sum()
    elif "statistiques" in question:
        return dataset.describe()
    else:
        return "Question non reconnue."

def nettoyage_transformation(dataset):
    dataset = dataset.copy()
    dataset = dataset.dropna()
    if 'sex' in dataset.columns:
        dataset['sex'] = dataset['sex'].map({'M': 1, 'F': 0})
    if 'physical_activity' in dataset.columns:
        dataset['physical_activity'] = dataset['physical_activity'].map({'Low': 0, 'Intermediate': 1, 'High': 2})
    return dataset

def normalisation_donnees(dataset, method="minmax"):
    dataset = dataset.copy()
    numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    dataset[numerical_cols] = scaler.fit_transform(dataset[numerical_cols])
    return dataset

def visualisation(dataset, variable_x, variable_y=None, plot_type="hist"):
    plt.figure(figsize=(8,5))
    if plot_type == "hist":
        sns.histplot(dataset[variable_x], kde=True)
    elif plot_type == "scatter" and variable_y:
        sns.scatterplot(x=dataset[variable_x], y=dataset[variable_y])
    elif plot_type == "box":
        sns.boxplot(x=dataset[variable_x])
    elif plot_type == "heatmap":
        sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
    else:
        return "Type de graphique non pris en charge."
    plt.title(f"Visualisation de {variable_x}")
    plt.show()

def charger_donnees(fichier, separateur=","):
    return pd.read_csv(fichier, sep=separateur)

def fusionner_donnees(df1, df2, cle):
    return df1.merge(df2, on=cle, how="left")
''')
if __name__ == "__main__":
    print("Module data_pipeline chargé avec succès.")