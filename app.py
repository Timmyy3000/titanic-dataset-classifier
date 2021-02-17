import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from PIL import Image

# get data
titanic_raw = pd.read_csv('train.csv')[['Survived', 'Pclass', 'Age', 'Sex']]
df = titanic_raw.head(5)


image = Image.open('Titanic.jpeg')

st.image(image, caption='The RMS Titanic',use_column_width=True)


st.write("""


# Titanic Classifier
------

This app classifies the **Survival** of RMS-Titanic passsengers!

Data obtained from the [Kaggle Titanic Dataset](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.

Find more about my submisison on [Kaggle](https://www.kaggle.com/timilehinogunme) 

""")

st.subheader('Feature Set')
st.write("""The models use 4 attributes to make a prediction on the probability of a passenger surviving.
        **Sample Feature Set**
         """)
st.write(df)
st.write('*Hint : Use these as a sample input parameters*')

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file]('example.csv')

""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():

        p_class = st.sidebar.selectbox('Passenger Class', ('1', '2', '3'))
        age = st.sidebar.slider('Age', 1, 100, 1)
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))

        data = {'Pclass': p_class,
                'Age': age,
                'Sex': sex,

                }
        features = pd.DataFrame(data, index=[0])
        return features

input_df = user_input_features()



# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using input parameters (shown below).')
    st.write(input_df)

st.subheader('Prediction')
st.write("""
You can select between a [*Logistic Regression*](https://en.wikipedia.org/wiki/Logistic_regression) model or a [*Random Forest Classifier*](https://en.wikipedia.org/wiki/Random_forest) to use as the classifier for this test.

Accuracy is calculated based on test data evaluation from Kaggle.com 
""")
model_type = st.selectbox('Choose Classifier Model', ('Logistic Regression, Accuracy : ~78%', 'Random Forest Classifier, Accuracy : ~72%'))

# Apply model to make predictions


st.subheader('Prediction')
# ENCODING SEX COLUMN
le = preprocessing.LabelEncoder()
input_df['Sex'] = le.fit_transform(input_df['Sex'])

if st.button('Predict'):
    if model_type == 'Logistic Regression, Accuracy : ~78%':
        # Reads in saved classification model
        load_clf = pickle.load(open('model-log.pickle', 'rb'))
    else:
        # Reads in saved classification model
        load_clf = pickle.load(open('model.pickle', 'rb'))

    prediction = load_clf.predict(input_df)
    prediction_proba = load_clf.predict_proba(input_df) * 100
    survival = np.array(['No', 'Yes'])
    st.write(pd.Series(survival[prediction], index=['Survived'], ))

    # Intepreting confidence level
    if prediction == 0:
        if prediction_proba[0][0] > 80:
            st.write("""
                ## Very Confident 
                """)
        elif prediction_proba[0][0] > 60:
            st.write("""
            ## Confident 
            """)
        else:
            st.write("""
            ## Unsure
            """)
    elif prediction == 1:
        if prediction_proba[0][1] > 80:
            st.write("""
                ## Very Confident 
                """)
        elif prediction_proba[0][1] > 60:
            st.write("""
            ## Confident 
            """)
        else:
            st.write("""
            ## Unsure
            """)
    st.subheader('Prediction Probability')
    st.write(pd.DataFrame(prediction_proba, columns=['No', 'Yes']))
