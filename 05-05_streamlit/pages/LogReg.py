import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogReg:
    def __init__(self, learning_rate, n_inputs):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs


    def fit(self, X, y):
        self.coef_ = np.random.normal(size=X.shape[1]) # w1..., wn
        self.intercept_ = np.random.normal() # w0
        X = np.array(X) # X - данные x1, x2, ..., xn
        y = np.array(y) # факт.показатель
        n_epoch = 10000
        for _ in range(n_epoch):
            z = np.dot(X, self.coef_) + self.intercept_ #  z = w1 * x1[i] + w2 * x2[i] + w0
            sigmoid_y_pred = 1 / (1 + np.exp(-z)) 
            error = y - sigmoid_y_pred   
            grad_coef = -(np.dot(X.T, error)) # -xi * (y_true - p), y_true - p = error
            grad_intercept = -error   # -(y[i] - sigmoid(yhat))
            self.coef_ -= self.learning_rate * grad_coef.mean(axis=0)
            self.intercept_ -= self.learning_rate * grad_intercept.mean()
     #   return self.coef_, self.intercept_   

    def predict(self, X):
        z = np.dot(X, self.coef_) + self.intercept_ # w1 * x1[i] + w2 * x2[i] + w0
        sigmoid_y_pred = 1 / (1 + np.exp(-z))
        return sigmoid_y_pred



st.title("Логистическая регрессия")
uploaded_train = st.file_uploader("Загрузите обучающую выборку CSV", type=["csv"])
uploaded_test = st.file_uploader("Загрузите тестовую выборку CSV", type=["csv"])
if uploaded_train is not None and uploaded_test is not None:
    train = pd.read_csv(uploaded_train)
    test = pd.read_csv(uploaded_test)
    features = st.multiselect("Выберите параметры для обучения", train.columns, default=[]) # сохраняем параметры для нормировки
    y_true = st.selectbox("Выберите параметр обучения (таргет)", train.columns)
    normalize = st.checkbox("Нормировать данные (StandardScaler)")
    
    if normalize:
        scaler = StandardScaler()
        train[features] = scaler.fit_transform(train[features])
        test[features] = scaler.fit_transform(test[features])    
    feature_numbers = len(features) # выбор кол-ва параметров, чтобы потом в классе назначить n_inputs
    learn_rate = st.slider("Скорость обучения", min_value=0.01, max_value=1.0, value=0.01) # выбор скорости, чтобы потом в классе применить# Начало обучения

    if st.button("Начать обучение"):
        model = LogReg(learn_rate, feature_numbers)  # вызываем класс
        model.fit(train[features], train[y_true]) # начинаем обучение
        train_coefs = model.coef_ # сохраняем веса, чтобы потом добавить в таблицу
        train_intercept = model.intercept_
        selected_features = ', '.join(features)
        st.write(f"Выбранные параметры: {selected_features}")
        st.write("Веса модели (коэффициенты):")
        st.write(pd.DataFrame(train_coefs, index=features, columns=["Weights"]))  
        
        train_pred = model.predict(train[features])
        train_comparison_df = pd.DataFrame({'Actual': train[y_true], 'Predicted': train_pred})
        st.write("Сравнение предсказанных и реальных значений на обучающем наборе данных:")
        st.write(train_comparison_df)

        test_pred = model.predict(test[features])
        test_comparison_df = pd.DataFrame({'Actual': test[y_true], 'Predicted': test_pred})
        st.write("Сравнение предсказанных и реальных значений на тестовом наборе данных:")
        st.write(test_comparison_df)

   
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.scatterplot(data=train, x=train[features[0]], y=train[features[1]], hue=train[y_true])
        x_range = np.linspace(train[features[0]].min(), train[features[0]].max(), 100)
        y_range = -(train_intercept + train_coefs[0] * x_range) / train_coefs[1]
        ax.plot(x_range, y_range, color='green')
        ax.set_xlabel(train.columns[0])
        ax.set_ylabel(train.columns[1])
        ax.set_title('Logistic Regression Line')
        ax.legend(['Logistic Regression Line', '0', '1'])
        st.pyplot(fig)
else:
    st.write("Пожалуйста, загрузите данные")