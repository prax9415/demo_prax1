
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle 
import streamlit as st

df=pd.read_csv('placement.csv') 
st.title("Scatterplot of CGPA and Package")
st.scatter_chart(df)

# plt.xlabel('CGPA')
# plt.ylabel('Package(in lpa)')

X=df.drop('package',axis=1)
y=df.package

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42) 
from sklearn.linear_model import LinearRegression
lr=LinearRegression() 
lr.fit(X_train,y_train)
y_train_pred = lr.predict(X_train) 
st.bar_chart(y_train_pred)
# plt.plot(X_train,lr.predict(X_train),color='red')
# plt.xlabel('cgpa')
# plt.ylabel("package")



