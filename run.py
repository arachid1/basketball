
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
from itertools import combinations

# Load the data
file_path = './mnt/data/DATAS.xlsx'  # Update this path to your actual file path
df = pd.read_excel(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.', '_')

# Identify categorical columns
categorical_columns = ['Nom', 'Prenom', 'Sexe', 'Oeil_directeur', 'Correction_visuel', 'Poste_jeu', 'Pas1_Cote_Horiz_fort', 'Pas1_CorrelOeilDirecteur_Lateralite', 'Pas1_Cote_Verti_Fort', 'Pas2_Cote_Horiz_fort', 'Pas2_CorrelOeilDirecteur_Lateralite', 'Pas2_Cote_Verti_Fort']
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
        model = ols(f'{col} ~ C(Poste_jeu)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_results[col] = anova_table['PR(>F)'][0]
    return anova_results

# Perform ANOVA
anova_results = perform_anova(df, df['Poste_jeu'].unique())
anova_df = pd.DataFrame(anova_results.items(), columns=['Variable', 'p_val'])

# Plotting the ANOVA results FLAT
plt.figure(figsize=(36, 24))
plt.barh(anova_df['Variable'], anova_df['p_val'], color='skyblue')
plt.axvline(x=0.05, color='r', linestyle='--')
plt.xlabel('p-value')
plt.title('ANOVA p-values for Numerical Columns across Positions')
plt.savefig('./mnt/data/anova_results.png')
# Plotting the ANOVA results STACKED
anova_results_pas1 = {k: v for k, v in anova_results.items() if k.startswith('Pas1_')}
anova_results_pas2 = {k.replace('Pas1_', 'Pas2_'): v for k, v in anova_results.items() if k.replace('Pas1_', 'Pas2_') in anova_results}
combined_results = pd.DataFrame({
    'Variable': list(anova_results_pas1.keys()),
    'Pas1_p_val': list(anova_results_pas1.values()),
    'Pas2_p_val': [anova_results_pas2.get(k.replace('Pas1_', 'Pas2_'), np.nan) for k in anova_results_pas1.keys()]
})
combined_results['Variable'] = combined_results['Variable'].str.replace('Pas1_', '')
plt.figure(figsize=(14, 8))
bar_width = 0.4
index = np.arange(len(combined_results))
plt.bar(index, combined_results['Pas1_p_val'], bar_width, label='Pas1', color='b')
plt.bar(index, combined_results['Pas2_p_val'], bar_width, bottom=combined_results['Pas1_p_val'], label='Pas2', color='g')
plt.xlabel('Variables')
plt.ylabel('p-value')
plt.title('ANOVA p-values for Pas1 and Pas2 Variables across Positions')
plt.xticks(index, combined_results['Variable'], rotation=90)
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold')
plt.legend()
plt.tight_layout()
plt.savefig('./mnt/data/anova_stacked_bars.png')
# Plotting the ANOVA results by POSITION
positions = df['Poste_jeu'].unique()
position_pairs = list(combinations(positions, 2))
for pos1, pos2 in position_pairs:
    df_position_pair = df[(df['Poste_jeu'] == pos1) | (df['Poste_jeu'] == pos2)]
    if df_position_pair['Poste_jeu'].nunique() < 2:
        print(f"Not enough samples for comparison between {label_encoders['Poste_jeu'].inverse_transform([pos1])[0]} and {label_encoders['Poste_jeu'].inverse_transform([pos2])[0]}. Skipping...")
        continue
    anova_results = perform_anova(df_position_pair, df_position_pair['Poste_jeu'].unique())
    anova_results_pas1 = {k: v for k, v in anova_results.items() if k.startswith('Pas1_')}
    anova_results_pas2 = {k.replace('Pas1_', 'Pas2_'): v for k, v in anova_results.items() if k.replace('Pas1_', 'Pas2_') in anova_results}
    combined_results = pd.DataFrame({
        'Variable': list(anova_results_pas1.keys()),
        'Pas1_p_val': list(anova_results_pas1.values()),
        'Pas2_p_val': [anova_results_pas2.get(k.replace('Pas1_', 'Pas2_'), np.nan) for k in anova_results_pas1.keys()]
    })
    combined_results['Variable'] = combined_results['Variable'].str.replace('Pas1_', '')
    plt.figure(figsize=(14, 8))
    bar_width = 0.4
    index = np.arange(len(combined_results))
    plt.bar(index, combined_results['Pas1_p_val'], bar_width, label='Pas1', color='b')
    plt.bar(index, combined_results['Pas2_p_val'], bar_width, bottom=combined_results['Pas1_p_val'], label='Pas2', color='g')
    pos1_name = label_encoders['Poste_jeu'].inverse_transform([pos1])[0]
    pos2_name = label_encoders['Poste_jeu'].inverse_transform([pos2])[0]
    plt.xlabel('Variables')
    plt.ylabel('p-value')
    plt.title(f'ANOVA p-values for Pas1 and Pas2 Variables: {pos1_name} vs {pos2_name}')
    plt.xticks(index, combined_results['Variable'], rotation=90)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./mnt/data/anova_stacked_bars_{pos1_name}_vs_{pos2_name}.png')


# Define a function to perform Chi-Square tests for categorical columns
def perform_chi_square(df, groups):
    chi2_results = {}
    for col in categorical_compare:
        contingency_table = pd.crosstab(df[col], df['Poste_jeu'])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        chi2_results[col] = p
    return chi2_results

# Perform Chi-Square tests
chi2_results = perform_chi_square(df, df['Poste_jeu'].unique())
chi2_df = pd.DataFrame(chi2_results.items(), columns=['Variable', 'p_val'])

# Plotting the Chi-Square results FLAT
plt.figure(figsize=(36, 24))
plt.barh(chi2_df['Variable'], chi2_df['p_val'], color='skyblue')
plt.axvline(x=0.05, color='r', linestyle='--')
plt.xlabel('p-value')
plt.title('Chi-Square p-values for Categorical Columns across Positions')
plt.savefig('./mnt/data/chi2_results.png')
# Plotting the Chi-Square results STACKED
chi2_results_pas1 = {k: v for k, v in chi2_results.items() if k.startswith('Pas1_')}
chi2_results_pas2 = {k.replace('Pas1_', 'Pas2_'): v for k, v in chi2_results.items() if k.replace('Pas1_', 'Pas2_') in chi2_results}
combined_results = pd.DataFrame({
    'Variable': list(chi2_results_pas1.keys()),
    'Pas1_p_val': list(chi2_results_pas1.values()),
    'Pas2_p_val': [chi2_results_pas2.get(k.replace('Pas1_', 'Pas2_'), np.nan) for k in chi2_results_pas1.keys()]
})
combined_results['Variable'] = combined_results['Variable'].str.replace('Pas1_', '')
plt.figure(figsize=(14, 8))
bar_width = 0.4
index = np.arange(len(combined_results))
plt.bar(index, combined_results['Pas1_p_val'], bar_width, label='Pas1', color='b')
plt.bar(index, combined_results['Pas2_p_val'], bar_width, bottom=combined_results['Pas1_p_val'], label='Pas2', color='g')
plt.xlabel('Variables')
plt.ylabel('p-value')
plt.title('Chi2 p-values for Pas1 and Pas2 Variables across Positions')
plt.xticks(index, combined_results['Variable'], rotation=90)
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold')
plt.legend()
plt.tight_layout()
plt.savefig('./mnt/data/chi2_stacked_bars.png')

# Plotting the Chi2 results by POSITION
positions = df['Poste_jeu'].unique()
position_pairs = list(combinations(positions, 2))
for pos1, pos2 in position_pairs:
    df_position_pair = df[(df['Poste_jeu'] == pos1) | (df['Poste_jeu'] == pos2)]
    min_group_size = df_position_pair['Poste_jeu'].value_counts().min()
    df_position_pair = df_position_pair.groupby('Poste_jeu').apply(lambda x: x.sample(n=min_group_size, random_state=42)).reset_index(drop=True)
    if df_position_pair['Poste_jeu'].nunique() < 2:
        pos1_name = label_encoders['Poste_jeu'].inverse_transform([pos1])[0]
        pos2_name = label_encoders['Poste_jeu'].inverse_transform([pos2])[0]
        print(f"Not enough samples for comparison between {pos1_name} and {pos2_name}. Skipping...")
        continue
    chi2_results = perform_chi_square(df_position_pair, df_position_pair['Poste_jeu'].unique())
    chi2_results_pas1 = {k: v for k, v in chi2_results.items() if k.startswith('Pas1_')}
    chi2_results_pas2 = {k.replace('Pas1_', 'Pas2_'): v for k, v in chi2_results.items() if k.replace('Pas1_', 'Pas2_') in chi2_results}
    combined_results = pd.DataFrame({
        'Variable': list(chi2_results_pas1.keys()),
        'Pas1_p_val': list(chi2_results_pas1.values()),
        'Pas2_p_val': [chi2_results_pas2.get(k.replace('Pas1_', 'Pas2_'), np.nan) for k in chi2_results_pas1.keys()]
    })
    combined_results['Variable'] = combined_results['Variable'].str.replace('Pas1_', '')
    plt.figure(figsize=(14, 8))
    bar_width = 0.4
    index = np.arange(len(combined_results))
    plt.bar(index, combined_results['Pas1_p_val'], bar_width, label='Pas1', color='b')
    plt.bar(index, combined_results['Pas2_p_val'], bar_width, bottom=combined_results['Pas1_p_val'], label='Pas2', color='g')
    pos1_name = label_encoders['Poste_jeu'].inverse_transform([pos1])[0]
    pos2_name = label_encoders['Poste_jeu'].inverse_transform([pos2])[0]
    plt.xlabel('Variables')
    plt.ylabel('p-value')
    plt.title(f'Chi-square p-values for Pas1 and Pas2 Variables: {pos1_name} vs {pos2_name}')
    plt.xticks(index, combined_results['Variable'], rotation=90)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./mnt/data/chi2_stacked_bars_{pos1_name}_vs_{pos2_name}.png')
    # plt.show()
################################ 

# Function to perform Friedman test
def perform_friedman_test(df):
    friedman_results = {}
    for col in numerical_compare:
        groups = [df[df['Poste_jeu'] == group][col].dropna().values for group in df['Poste_jeu'].unique()]
        min_size = min(len(group) for group in groups)
        trimmed_groups = [group[:min_size] for group in groups]
        
        # Check if all groups have at least one observation
        if all(len(group) > 0 for group in trimmed_groups):
            stat, p = stats.friedmanchisquare(*trimmed_groups)
            friedman_results[col] = p
        else:
            print(f"Skipping {col} due to insufficient data after trimming.")
    return friedman_results

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

# Perform Friedman test
friedman_results = perform_friedman_test(df)
friedman_df = pd.DataFrame(friedman_results.items(), columns=['Variable', 'p_val'])

# Plotting the Friedman test results
plt.figure(figsize=(12, 8))
plt.barh(friedman_df['Variable'], friedman_df['p_val'], color='skyblue')
plt.axvline(x=0.05, color='r', linestyle='--')
plt.xlabel('p-value')
plt.title('Friedman Test p-values for Numerical Columns across Positions')
plt.savefig('./mnt/data/friedman_results.png')


# Save results to CSV and HTML files
anova_df.to_csv('./mnt/data/anova_results.csv')
chi2_df.to_csv('./mnt/data/chi2_results.csv')
friedman_df.to_csv('./mnt/data/friedman_results.csv')

# Save Tukey's HSD test results
with open('./mnt/data/tukey_results.html', 'w') as f:
    for col, result in tukey_df.items():
        f.write(f'<h2>{col}</h2>')
        f.write(result)

# Summary printout
print("Analysis complete. Results saved to ./mnt/data/")
