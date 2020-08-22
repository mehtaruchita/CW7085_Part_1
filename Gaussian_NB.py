#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
df = pd.read_csv(r'C:\Ruchita\MSc_Data_Science\Module-8-7085-Advanced_MAchine_Learning\CW-7085\Part_1\Autism\autism_output.csv')
df.head(10)
df.describe()
df.dtypes
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df = clean_dataset(df)
X = df.iloc[:,0:18]
Y = df.iloc[:,18]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.20,random_state=88)
#Training and fitting Gaussian Naive Bayes Model
Gauss_NB = GaussianNB()
Gauss_NB.fit(X_train,Y_train)
pred_rf = Gauss_NB.predict(X_test)
print('Accuracy Score:',accuracy_score(Y_test,pred_rf))
print('Confusion Matrix: \n',confusion_matrix(Y_test,pred_rf))
print('Classification Report: \n',classification_report(Y_test,pred_rf))


