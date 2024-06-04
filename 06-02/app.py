import joblib
import pandas as pd
import streamlit as st


# Важная настройка для корректной настройки pipeline!
import sklearn
sklearn.set_config(transform_output="pandas")


ml_pipline = joblib.load('06_02_ml_pipeline_voting.pkl')

st.title('Модель предсказания сердечных заболеваний')
st.subheader('Выберите значения параметров')
Age = st.number_input("Введите ваш возраст", min_value=0, max_value=100, step=1, value=0, format="%d")
Sex = st.selectbox("Выберите пол", ["M", "F"])
st.write('M - мужской, F - женский')
ChestPainType = st.selectbox("Выберите тип боли", ["TA", "ATA", "NAP", "ASY"])
st.write('TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomati')
RestingBP = st.number_input("Введите кровяное давление", min_value=10, max_value=300, step=1, value=10, format="%d")
Cholesterol = st.number_input("Введите уровень холестерина", min_value=10, max_value=300, step=1, value=10, format="%d")
FastingBS = st.selectbox("Если уровень сахара более 120 mg/dl, выберите 1. Если нет, то 0", [1, 0])
RestingECG = st.selectbox("Результат ЭКГ", ["Normal", "ST", "LVH"])
st.write("Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite * left ventricular hypertrophy by Estes criteria")
MaxHR = st.number_input("Давление", min_value=60, max_value=202, step=1, value=60, format="%d") 
ExerciseAngina = st.selectbox("Есть ли стенокардия, вызванная физической нагрузкой", ['Y', 'N'])
st.write('Y - да, N - нет')
Oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1, value=0.0,format='%.2f')
ST_Slope = st.selectbox("the slope of the peak exercise ST segment", ["Up", "Flat", "Down"])
st.write("Up: upsloping, Flat: flat, Down: downsloping")

df = pd.DataFrame({'Age': [Age], 'Sex': [Sex], 'ChestPainType': [ChestPainType], 'RestingBP': [RestingBP], 'Cholesterol': [Cholesterol], 'FastingBS': [FastingBS], 
                   'RestingECG': [RestingECG], 'MaxHR': [MaxHR], 'ExerciseAngina': [ExerciseAngina], 'Oldpeak': [Oldpeak], 'ST_Slope': [ST_Slope]})

st.table(df)
y_pred = ml_pipline.predict(df)
st.write("Прогноз наличия сердечных заболеваний: ", y_pred)
st.write("1 - есть, 0 - нет")