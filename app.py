import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st 

dataset=pd.read_csv('crop.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

classifier=LogisticRegression()
classifier.fit(X_train,Y_train)

st.title('Crop Recommendation')
n=st.number_input('Enter Nitogen:')
p=st.number_input('Enter Phosphorus:')
k=st.number_input('Enter Potassium:')
t=st.number_input('Enter Temperature:')
h=st.number_input('Enter Humidity:')
ph=st.number_input('Enter pH:')
r=st.number_input('Enter rainfall:')

if st.button('Recommend Crop'):
    data=[[n,p,k,t,h,ph,r]]
    result=classifier.predict(data)[0]
    st.success(result)