
# Voila les Datas de mon expérience. 
# Dans les premiere colonnes y'a les info pour identifier le joueurs qui a fait le test [A-E].
# Nom
# Prénom
# Sexe
# Date de naissance
# Age
# Puis des infos sur sa vue [F-G].
# Oeil directeur
# Correction visuel
# Ensuite des informations sur son vecu basket [H-L].
# Année de pratique de basketball
# Nombre d'entrainement par semaine
# Meilleur niveau jouer en compétition
# Niveau jouer actuellement
# Poste de jeu 
# On a ensuite des données sur chaque passage [M-Y] pour le 1 et [Z-AL] pour le passage 2. Dans l'ordre pour chacun on a chaque fois : 
# Fatigue avant
# Fatigue apres 
# Différence de fatigue
# Temps de réaction total moyen
# Temps de réaction Zone visuel supérieur moyen
# Temps de réaction Zone visuel Inférieur moyen
# Temps de réaction Zone visuel gauche moyen
# Temps de réaction Zone visuel Droit moyen
# Ecart gauche droite (droite-gauche)
# Ecart Haut bas (Bas-Haut)
# Déduction préférence attentionnel horizontal/latéral
# Correlation avec l'oeil directeur ou non
# Déduction préférence attentionnel vertical
# Ensuite  comparaison entre passages avec a chaque fois si il y'a eu une amélioration entre le passage 1 et 2
# Réaction total
# Réaction supérieur
# Réaction inférieur
# Réaction Gauche
# Réaction Droit
# Et apres on a les données brut [AW-FL]
# A chaque fois j'ai noté par exemple Pas1_318 qui signifie que c'est le premiere passage pour pas1 et le 318 est l'angle d'arrivé de la balle. 

# Si tu as des questions, n'hésite pas.
# D'abord, je veux comparer s'il y a une différence significative entre le passage 1 et le passage 2 pour tous les participants. Ça, ce n'est pas compliqué, je peux le faire. Ce que je n'arrive pas à faire, c'est par exemple comparer un groupe de personnes à part ou bien entre eux. Par exemple j'aimerais comparer s'il y a une différence significative entre les meneurs et les pivots dans le passage 1 et le passage 2.

# J'aimerais faire cela avec différentes données qualitatives, comme par exemple leur œil directeur. En fait, l'idée, c'est de pousser le truc et de comparer des groupes spécifiques comme les meneurs avec l'œil directeur gauche.
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import itertools
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

# Load the data
file_path = './mnt/data/DATAS.xlsx'  # Update this path to your actual file path
df = pd.read_excel(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Identify categorical columns
categorical_columns = ['Nom', 'Prenom', 'Sexe', 'Oeil_directeur', 'Correction_visuel', 'Poste_jeu', 'Pas1_Cote_Horiz_fort', 'Pas1_CorrelOeilDirecteur_Lateralite', 'Pas1_Cote_Verti_Fort', 'Pas2_Cote_Horiz_fort', 'Pas1_CorrelOeilDirecteur_Lateralite', 'Pas2_Cote_Verti_Fort']
columns_to_compare = [col for col in df.columns if col.startswith('Pas1_') or col.startswith('Pas2_')]

# Encode categorical columns using Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate categorical and numerical columns among those to compare
categorical_compare = [col for col in columns_to_compare if col in categorical_columns]
numerical_compare = [col for col in columns_to_compare if col not in categorical_columns]

# Define a function to perform ANOVA for numerical columns
def perform_anova(df, groups):
    anova_results = {}
    for col in numerical_compare:
        print(col)
        model = ols(f'{col} ~ C(Poste_jeu)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_results[col] = anova_table['PR(>F)'][0]
    return anova_results

# Define a function to perform Chi-Square tests for categorical columns
def perform_chi_square(df, groups):
    chi2_results = {}
    for col in categorical_compare:
        contingency_table = pd.crosstab(df[col], df['Poste_jeu'])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        chi2_results[col] = p
    return chi2_results

# Perform ANOVA
anova_results = perform_anova(df, df['Poste_jeu'].unique())
anova_df = pd.DataFrame(anova_results.items(), columns=['Variable', 'p_val'])

# Perform Chi-Square tests
chi2_results = perform_chi_square(df, df['Poste_jeu'].unique())
chi2_df = pd.DataFrame(chi2_results.items(), columns=['Variable', 'p_val'])

# Plotting the ANOVA results
plt.figure(figsize=(12, 8))
plt.barh(anova_df['Variable'], anova_df['p_val'], color='skyblue')
plt.axvline(x=0.05, color='r', linestyle='--')
plt.xlabel('p-value')
plt.title('ANOVA p-values for Numerical Columns across Positions')
plt.savefig('/mnt/data/anova_results.png')

# Plotting the Chi-Square results
plt.figure(figsize=(12, 8))
plt.barh(chi2_df['Variable'], chi2_df['p_val'], color='skyblue')
plt.axvline(x=0.05, color='r', linestyle='--')
plt.xlabel('p-value')
plt.title('Chi-Square p-values for Categorical Columns across Positions')
plt.savefig('/mnt/data/chi2_results.png')

# Function to perform Friedman test
def perform_friedman_test(df):
    friedman_results = {}
    for col in numerical_compare:
        groups = [df[df['Poste_jeu'] == group][col].dropna().values for group in df['Poste_jeu'].unique()]
        stat, p = stats.friedmanchisquare(*groups)
        friedman_results[col] = p
    return friedman_results

# Perform Friedman test
friedman_results = perform_friedman_test(df)
friedman_df = pd.DataFrame(friedman_results.items(), columns=['Variable', 'p_val'])

# Plotting the Friedman test results
plt.figure(figsize=(12, 8))
plt.barh(friedman_df['Variable'], friedman_df['p_val'], color='skyblue')
plt.axvline(x=0.05, color='r', linestyle='--')
plt.xlabel('p-value')
plt.title('Friedman Test p-values for Numerical Columns across Positions')
plt.savefig('/mnt/data/friedman_results.png')

# Perform Tukey's HSD test for pairwise comparisons within groups for numerical columns
def perform_tukey(df):
    tukey_results = {}
    for col in numerical_compare:
        tukey = pairwise_tukeyhsd(endog=df[col].dropna(), groups=df['Poste_jeu'].dropna(), alpha=0.05)
        tukey_results[col] = tukey.summary()
    return tukey_results

# Perform Tukey's HSD test
tukey_results = perform_tukey(df)
tukey_df = {col: tukey_results[col].as_html() for col in tukey_results}

# Save results to CSV and HTML files
anova_df.to_csv('/mnt/data/anova_results.csv')
chi2_df.to_csv('/mnt/data/chi2_results.csv')
friedman_df.to_csv('/mnt/data/friedman_results.csv')

# Save Tukey's HSD test results
with open('/mnt/data/tukey_results.html', 'w') as f:
    for col, result in tukey_df.items():
        f.write(f'<h2>{col}</h2>')
        f.write(result)

# Summary printout
print("Analysis complete. Results saved to /mnt/data/")
