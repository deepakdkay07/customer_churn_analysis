# Importing
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# The following settings will improve the default style and font sizes for our charts.  
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']=(10,6)
matplotlib.rcParams['figure.facecolor']='#00000000'

def accuracy_heatmap(pipeline):
    all_cols=pipeline.named_steps['preprocessor'].get_feature_names_out()
    acc=pipeline.score(X_test,y_test)
    st.write(f"The Accuracy on Test Dataset with LogisticRegression Model is : {acc:.2%}")
    test_predict=pipeline.predict(X_test)
    cf = confusion_matrix(y_test, test_predict,normalize='true')    
    fig, ax = plt.subplots(figsize=(5,4))   
    sns.heatmap(
        cf,
        annot=True,
        cmap='Blues',
        ax=ax
    )   
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')    
    st.pyplot(fig)
    if pipeline in [pipeline1,pipeline2_ht,pipeline2,pipeline2_ht]:
        rf=pipeline.named_steps['model']
        importance = pd.DataFrame({
            'feature': list(all_cols),
            'importance': pipeline.named_steps['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        st.dataframe(importance)

def score_(pipeline):
    sc=pipeline.score(X_test,y_test)
    return (f"{sc:.2%}")

def single_predict(single):
    st.write('For SingleValueDataSet')
    st.dataframe(single)
    pipeline = joblib.load('LogisticRegression.joblib')
    pipeline1 = joblib.load("DecisionTreeClassifier.joblib")
    pipeline2 = joblib.load("RandomForestClassifier.joblib")

    # Tuned DecisionTreeClassifier
    pipeline1_ht=joblib.load("DecisionTreeClassifier_ht.joblib")

    # Tuned RandomForestClassifier
    pipeline2_ht=joblib.load("RandomForestClassifier_ht.joblib")

    input_df = pd.DataFrame([single])
    st.write(f'LogisticRegression Prediction : {pipeline.predict(input_df)},with Probability : {pipeline.predict_proba(input_df)[0][0]*100:.2f}%')
    st.write(f'DecisionTree Prediction : {pipeline1.predict(input_df)},with Probability : {pipeline1.predict_proba(input_df)[0][0]*100:.2f}%')
    st.write(f'DecisionTree Tuned Prediction : {pipeline1_ht.predict(input_df)},with Probability : {pipeline1_ht.predict_proba(input_df)[0][0]*100:.2f}%')
    st.write(f'RandomForest Prediction : {pipeline2.predict(input_df)},with Probability : {pipeline2.predict_proba(input_df)[0][0]*100:.2f}%')
    st.write(f'RandomForest Tuned Prediction : {pipeline2_ht.predict(input_df)},with Probability : {pipeline2_ht.predict_proba(input_df)[0][0]*100:.2f}%')

st.title("Project Analysis")
choose=st.segmented_control("Choose",["Churn Dataset Analysis","Single Value Input"])
churn_df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


if choose == "Churn Dataset Analysis":
    st.write('Top 10 Values in Dataset :')
    st.write(churn_df.head(10))
    churn_df['TotalCharges']=pd.to_numeric(churn_df['TotalCharges'],errors='coerce')

    # Defining Input and Output
    X = churn_df.drop(['customerID','Churn'],axis=1)
    y=churn_df['Churn']

    # EDA and Model in pipeline
    numeric_cols=X.select_dtypes(np.number).columns.tolist()
    categorical_cols=X.select_dtypes('object').columns.tolist()

    # Test Split
    ratio = st.select_slider("Any Percentage of Ratio You would Like for train and test split?",[i*0.01 for i in range(20,85,5)],key="level_slider")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=ratio,random_state=42) 
    st.write(f"Splitting {ratio}% Dataset for testing models")

    # Importing Saved Models so they wont train again and again
    pipeline = joblib.load('LogisticRegression.joblib')
    pipeline1 = joblib.load("DecisionTreeClassifier.joblib")
    pipeline2 = joblib.load("RandomForestClassifier.joblib")

    # Tuned DecisionTreeClassifier
    pipeline1_ht=joblib.load("DecisionTreeClassifier_ht.joblib")

    # Tuned RandomForestClassifier
    pipeline2_ht=joblib.load("RandomForestClassifier_ht.joblib")

    # Every model Evaluations
    # Which Model analysis You would like to see
    st.write(f'Note : Models are HyperTuned when there is 20% split of test data')
    mdl=st.segmented_control("Any specific Model you would Like to See",["LogisticRegression","DecisionTreeClassifier","DecisionTreeClassifier_HyperTuned","RandomForestClassifier","RandomForestClassifier_HyperTuned","All_without_heatmap_Graph"])
    map_={"LogisticRegression":pipeline,"DecisionTreeClassifier":pipeline1,"DecisionTreeClassifier_HyperTuned":pipeline1_ht,"RandomForestClassifier":pipeline2,"RandomForestClassifier_HyperTuned":pipeline2_ht}
    if mdl == "All_without_heatmap_Graph":
        scores = pd.DataFrame({
            'Model':['Logistic Regression','Decision Tree','Decision Tree Tuned','Random Forest','Random Forest Tuned',],
            'Accuracy':[score_(pipeline),score_(pipeline1),score_(pipeline1_ht),score_(pipeline2),score_(pipeline2_ht)]
        })
        st.dataframe(scores)

    elif mdl in ["LogisticRegression","DecisionTreeClassifier","DecisionTreeClassifier_HyperTuned","RandomForestClassifier","RandomForestClassifier_HyperTuned"]:
        #print(map_[mdl])
        accuracy_heatmap(map_[mdl])


elif choose == "Single Value Input" :
    single = {'gender':st.segmented_control('Gender',['Male','Female']), 'SeniorCitizen':st.segmented_control('SeniorCitizen',[0,1]),
        'Partner':st.segmented_control('Partner',['Yes','No']),'Dependents':st.segmented_control('Dependents',['Yes','No']),
        'tenure':st.number_input('Tenure in range 0 to 72',min_value=0,max_value=72),'PhoneService':st.segmented_control('PhoneServices',['Yes','No']), 'MultipleLines':st.segmented_control('MultipleLines',['Yes','No','No phone service']),
        'InternetService':'DSL', 'OnlineSecurity':st.segmented_control('OnlineSecurity',['Yes','No','No internet service']),
        'OnlineBackup':st.segmented_control('OnlineBackup',['Yes','No','No internet service']), 'DeviceProtection':st.segmented_control('DeviceProtection',['Yes','No','No internet service']),
        'TechSupport':st.segmented_control('TechSupport',['Yes','No','No internet service']), 'StreamingTV':st.segmented_control('StreamingTV',['Yes','No','No internet service']),
        'StreamingMovies':st.segmented_control('StreamingMovies',['Yes','No','No internet service']), 'Contract':'One year', 'PaperlessBilling':st.segmented_control('PaperlessBiling',['Yes','No']), 'PaymentMethod':'Bank transfer (automatic)',
        'MonthlyCharges':st.number_input('MonthlyCharges in range 18.25 to 118.75',min_value=18.25,max_value=118.72), 'TotalCharges':st.number_input('TotalCharges in range 18.8 to 8684.8',min_value=18.8,max_value=8684.8)}
    single_predict(single)