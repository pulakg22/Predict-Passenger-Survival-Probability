import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('titanic_data.csv')

# Preprocess data
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna('S', inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Feature scaling
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Split data
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Classifier': SVC(probability=True)
}

for model_name, model in models.items():
    model.fit(X, y)

# Streamlit app
st.title('Titanic Survival Prediction')
st.write("Enter the passenger's details to predict their survival on the Titanic:")

# Model selection
selected_model_name = st.selectbox('Select Model', list(models.keys()))
selected_model = models[selected_model_name]

# Input features from the user
pclass = st.selectbox('Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)', [1, 2, 3])
sex = st.selectbox('Sex (0 = Male; 1 = Female)', [0, 1])
age = st.slider('Age', 0, 80, 29)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=8, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=6, value=0)
fare = st.number_input('Fare', min_value=0.0, max_value=512.329, value=32.2)
embarked = st.selectbox('Port of Embarkation (0 = Southampton; 1 = Cherbourg; 2 = Queenstown)', [0, 1, 2])

# Create a prediction button
if st.button('Predict Survival'):
    # Scale user inputs
    scaled_age = scaler.transform([[age, fare]])[0, 0]
    scaled_fare = scaler.transform([[age, fare]])[0, 1]

    # Prepare the feature vector for prediction
    input_data = np.array([[pclass, sex, scaled_age, sibsp, parch, scaled_fare, embarked]])

    # Predict survival
    prediction = selected_model.predict(input_data)
    prediction_prob = selected_model.predict_proba(input_data)[0][1]

    # Output the prediction
    if prediction == 1:
        st.success(f"The passenger is likely to survive with a probability of {prediction_prob:.2f}.")
    else:
        st.error(f"The passenger is unlikely to survive with a probability of {1 - prediction_prob:.2f}.")
