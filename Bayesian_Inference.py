# Importing Libraries
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
from matplotlib import MatplotlibDeprecationWarning
from Util_file import display_probs
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 22
# %matplotlib inline
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)
# dataset to dataframe
df = pd.read_csv(
    r'C:\Ruchita\MSc_Data_Science\Module-8-7085-Advanced_MAchine_Learning\CW-7085\Part_1\Autism\autism_output.csv')
# Clean dataset
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
df = clean_dataset(df)
X = df.iloc[:, 0:18]
Y = df.iloc[:, 18]
###########################################################
# Function for plotting Trace
def plot_traces(traces, retain=0):
    ax = pm.traceplot(traces[-retain:],
                      lines=tuple([(k, {}, v['mean'])
                                   for k, v in pm.summary(traces[-retain:]).iterrows()]))
    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
        ax[i, 0].annotate('{:.2f}'.format(mn), xy=(mn, 0), xycoords='data'
                          , xytext=(5, 5), textcoords='offset points', rotation=90
                          , va='bottom', fontsize='medium', color='#AA0022')
    plt.show()
#######################################################################
# Calculating Priors
ASD = ['0', '1']
c = np.array([1, 0])
alphas = np.array([1, 1])
alpha_list = [np.array([0.1, 0.1]), np.array([1, 1]),
              np.array([5, 5]), np.array([15, 15])]
display_probs(dict(zip(ASD, (alphas + c) / (c.sum() + alphas.sum()))))
display_probs(dict(zip(ASD, (203 / 703, 500 / 703))))
values = []
for alpha_new in alpha_list:
    values.append((alpha_new + c) / (c.sum() + alpha_new.sum()))

value_df = pd.DataFrame(values, columns=ASD)
value_df['alphas'] = [str(x) for x in alpha_list]
value_df
##################################################
# Plotting Priors
melted = pd.melt(value_df, id_vars='alphas', value_name='prevalence',
                 var_name='ASD')

plt.figure(figsize=(10, 10))
sns.barplot(x='alphas', y='prevalence', hue='ASD', data=melted,
            edgecolor='k', linewidth=1.0)
plt.xticks(size=12);
plt.yticks(size=12)
plt.title('Expected Value')
##############################################################
# Maximim Posterior Probabilities
display_probs(dict(zip(ASD, (alphas + c) / sum(alphas + c))))
values = []
for alpha_new in alpha_list:
    values.append((alpha_new + c - 1) / sum(alpha_new + c - 1))

value_df = pd.DataFrame(values, columns=ASD)
value_df['alphas'] = [str(x) for x in alpha_list]
value_df
melted = pd.melt(value_df, id_vars='alphas', value_name='prevalence',
                 var_name='ASD')

plt.figure(figsize=(8, 6))
sns.barplot(x='alphas', y='prevalence', hue='ASD', data=melted,
            edgecolor='k', linewidth=1.5)
plt.xticks(size=14);
plt.yticks(size=14)
plt.title('Maximum A Posterior Value')
niter = 1000
# PyMC3 Model GLM
with pm.Model() as logistic_model:
    pm.glm.GLM.from_formula('Y ~ X.iloc[:,17]+ X.iloc[:,11] +X.iloc[:,12] ',
                            df,
                            family=pm.glm.families.Normal())
    trace = pm.sample(500, tune=3000, init='adapt_diag', cores=1, target_accept=0.95)
summary = pm.summary(trace)
print('summary', summary)
