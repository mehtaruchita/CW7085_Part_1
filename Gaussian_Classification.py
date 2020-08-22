#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#Data Visualization
df_input = pd.read_csv(r'C:\Ruchita\MSc_Data_Science\Module-8-7085-Advanced_MAchine_Learning\CW-7085\Part_1\Autism\autism_input.csv')

plt.figure(figsize=(10,10))
corr_in = df_input.corr()
sns.heatmap(corr_in,mask=np.zeros_like(corr_in,dtype=np.bool),
            cmap=sns.diverging_palette(-100,0,as_cmap=True), square=True)
df_input['ASD'].value_counts().plot(kind = 'bar')
df_input['ethnicity'].value_counts().plot(kind = 'pie')
df_input['gender'].value_counts().plot(kind = 'bar')
df_input['jaundice'].value_counts().plot(kind = 'bar')
plt.figure(figsize=(15,7))
sns.countplot(df_input['ethnicity'],hue=df_input['ASD'])
sns.countplot(df_input['gender'],hue=df_input['ASD'])
sns.countplot(df_input['jaundice'],hue=df_input['ASD'])
#Description of preprocessed data
df = pd.read_csv(r'C:\Ruchita\MSc_Data_Science\Module-8-7085-Advanced_MAchine_Learning\CW-7085\Part_1\Autism\autism_output.csv')
df.head(10)
df.describe()
#Cleaning Dataset
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df = clean_dataset(df)
#Correlation plot of outout file
plt.figure(figsize=(10,10))
corr = df.corr()
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),
            cmap=sns.diverging_palette(-100,0,as_cmap=True), square=True)
X = df.iloc[:,0:18]
Y = df.iloc[:,18]
df.isnull().any()
df = df.fillna(lambda x: x.median())
#Test-Train Split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.20,random_state=88)
gp_tr = GaussianProcessClassifier()
gp_tr.fit(X_train,Y_train)
pred_tr = gp_tr.predict(X_test)
print("Log Marginal Likelihood : %.3f"
      % gp_tr.log_marginal_likelihood(gp_tr.kernel_.theta))
print('Confusion Matrix without Optimizing:', confusion_matrix(Y_test,pred_tr))
print('Accuracy Score:', accuracy_score(Y_test,pred_tr))
print('Classification Report: \n',classification_report(Y_test,pred_tr))



