#!/usr/bin/env python

import streamlit as st
# st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data and models
train_data = pd.read_csv("data_summary_data.csv")

with open("logr_model.pkl", "rb") as f:
    logr_model = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("logr_std_model.pkl", "rb") as f:
    logr_std_model = pickle.load(f)

with open("data_pipe.pkl","rb") as f:
    data_pipe=pickle.load(f)

cols=['Gender', 'Age', 'Neighbourhood', 'Scholarship',
            'Hipertension','Diabetes', 'Alcoholism', 'Handicap', 'SMS_received',
            'ScheduledDay_Month', 'ScheduledDay_Day_Of_Month',
            'ScheduledDay_Day_Of_Week', 'AppointmentDay_Month',
            'AppointmentDay_Day_Of_Month', 'AppointmentDay_Day_Of_Week']

# Sidebar Panel

st.sidebar.title("Options")
section = st.sidebar.radio("Choose Section", ["Data Summary", "Prediction", "Feature Importance"])

### 1. Data Summary Section

if section == "Data Summary":
    st.header("Data Summary For Numeric Features")
    
    # Display entire dataset
    # if st.sidebar.checkbox("Show Full Dataset"):
    #     st.write(train_data)

    # Display data summary
    if st.sidebar.checkbox("Show Data Summary"):
        st.write(train_data[cols].describe())

    # Visualization options
    feature_selection = st.sidebar.multiselect("Select Feature(s) for Visualization", cols)

    if len(feature_selection) == 1:
        feature = feature_selection[0]
        if feature in ['Age']:
            st.write(f"Distribution of {feature}")
            sns.histplot(train_data[feature], kde=True)
        else:
            st.write(f"Bar chart of {feature}")
            sns.set_style('whitegrid')
            my_palette=["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a", "#ff7c43", "#ffa600", "#58508d"]
            cat_order=list(train_data[feature].value_counts().index)
            sns.catplot(x=feature, data=train_data,kind='count',order=cat_order,palette=my_palette)
            plt.xticks(rotation=45,rotation_mode='anchor',ha='right',fontsize=3)
        st.pyplot()
    
    elif len(feature_selection) == 2:
        feature1, feature2 = feature_selection

        # Both features are numeric
        # if pd.api.types.is_numeric_dtype(train_data[feature1]) and pd.api.types.is_numeric_dtype(train_data[feature2]):
        #     st.write(f"Scatter plot between {feature1} and {feature2}")
        #     sns.scatterplot(x=feature1, y=feature2, data=train_data)
        
        # One feature is categorical, the other is numeric
        if  feature2 in ['Age']:
            st.write(f"Box plot of {feature2} grouped by {feature1}")
            sns.boxplot(x=feature1, y=feature2, data=train_data)
        elif feature1 in ['Age']:
            st.write(f"Box plot of {feature1} grouped by {feature2}")
            sns.boxplot(x=feature2, y=feature1, data=train_data)
        
        # Both features are categorical
        elif  (feature1 not in ['Age']) and (feature2 not in ['Age']):
            st.write(f"Heatmap of counts for {feature1} and {feature2}")
            pivot_table = pd.crosstab(train_data[feature1], train_data[feature2])
            sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu")

        else:
            st.write("Invalid combination for visualization")
        
        st.pyplot()

### 2. Prediction Section
elif section == "Prediction":
    st.header("Make a Prediction")

    # Collect input for prediction
    input_data = {}
    for col in ['Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'No-show']:  # assuming the last column is the target
        if col in ['Age']:
            
            input_data[col] = st.slider(
                f"Enter value for {col} ",
                min_value=2,
                max_value=98,
                value=30  # default value
            )
        else:
            input_data[col] = st.selectbox(f"Select value for {col}", train_data[col].unique())
    
    model_choice = st.sidebar.selectbox("Choose Model for Prediction", ["Logistic Regression", "Random Forest"])
    model = logr_model if model_choice == "Logistic Regression" else rf_model
    input_df = pd.DataFrame([input_data])
    for col in ['Scholarship', 'Hypertension','Diabetes', 'Alcoholism', 'SMS_received']:
    
        input_df[col]=input_df[col].replace({'No':0,'Yes':1})
        
    x=data_pipe.transform(input_df)

    if st.button("Predict"):
        prediction = model.predict_proba(x)[0]
        st.write(f"Prediction: {dict(zip(model.classes_,prediction))}")
        

        # # LIME Explanation
        # explainer = LimeTabularExplainer(x.values,
        #                                  feature_names=x.columns,
        #                                  class_names=['No Show'],
        #                                  discretize_continuous=True)
        # explanation = explainer.explain_instance(x.values[0], model.predict_proba)
        # st.write("Local Feature Importances")
        # st.write(explanation.as_list())
        # explanation.show_in_notebook(show_table=True)

### 3. Feature Importance Section
elif section == "Feature Importance":
    st.header("Global Model Interpretations")

    model_choice = st.sidebar.selectbox("Choose Model for Interpretation", ["Logistic Regression", "Random Forest"])
    model = logr_std_model if model_choice == "Logistic Regression" else rf_model
    x_train=pd.read_csv('x_train.csv')

    # Feature Importance
    if model_choice == "Random Forest":
        st.subheader("Feature Importance - Random Forest")
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': x_train.columns, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        st.pyplot()

    elif model_choice == "Logistic Regression":
        st.subheader("Feature Importance - Logistic Regression")
        coefficients = model.coef_[0]
        feature_importance_df = pd.DataFrame({'feature': x_train.columns, 'coefficient': coefficients})
        feature_importance_df = feature_importance_df.sort_values(by="coefficient", ascending=False)
        sns.barplot(x='coefficient', y='feature', data=feature_importance_df)
        st.pyplot()

    # Partial Dependence Plots (Placeholder, requires model-specific implementation)
    if st.sidebar.checkbox("Show Partial Dependence Plots"):
        st.write("Partial Dependence Plots (currently not implemented)")
        # Implement PDP here if desired