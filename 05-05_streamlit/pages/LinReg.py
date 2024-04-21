import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinReg:
    def __init__(self, learning_rate, n_inputs):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.coef_ = None
        self.intercept_ = None    
        
    def fit(self, X, y):
        X = np.array(X) # X - данные x1, x2, ..., xn
        y = np.array(y)
        self.coef_ = np.random.normal(size=X.shape[1]) # w1..., wn
        self.intercept_ = np.random.normal() # w0
        n_samples = X.shape[0]  # Получаем количество образцов в данных
        for _ in range(10000):
            y_pred = np.dot(X, self.coef_) + self.intercept_ # y_pred = x1 * w1 + w0
            error = y_pred - y
            grad_coef = (2 / n_samples) * np.dot(X.T, error)
            grad_intercept = (2 / n_samples) * np.sum(error)
            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept
       # return self.coef_, self.intercept_    
    def predict(self, X):
        y_pred = np.dot(X, self.coef_) + self.intercept_ # y_pred = x1 * w1 + w0
        return y_pred
    def score(self, X, y):
        y_pred = np.dot(X, self.coef_) + self.intercept_
        mse = np.mean((y - y_pred) ** 2)
        return mse

st.title("Линейная регрессия")
uploaded_train = st.file_uploader("Загрузите обучающую выборку CSV", type=["csv"])
uploaded_test = st.file_uploader("Загрузите тестовую выборку CSV", type=["csv"])
#model = None
if uploaded_train is not None and uploaded_test is not None:
    train = pd.read_csv(uploaded_train)
    test = pd.read_csv(uploaded_test)
    features = st.multiselect("Выберите параметры для обучения", train.columns, default=[]) # сохраняем параметры для нормировки
    y_true = st.selectbox("Выберите таргет", train.columns) # сохраняем таргет
    normalize = st.checkbox("Нормировать данные (StandardScaler)")    

    if normalize:
        scaler = StandardScaler()
        train[features] = scaler.fit_transform(train[features])
        test[features] = scaler.fit_transform(test[features])  

    feature_numbers = len(features) # выбор кол-ва параметров, чтобы потом в классе назначить n_inputs
    learn_rate = st.slider("Скорость обучения", min_value=0.01, max_value=1.0, value=0.01) # выбор скорости, чтобы потом в классе применить# Начало обучения# Начало обучения
    if st.button("Начать обучение"):
        model = LinReg(learn_rate, feature_numbers)  # вызываем класс
        model.fit(train[features], train[y_true]) # начинаем обучение
        train_coefs = model.coef_ # сохраняем веса, чтобы потом добавить в таблицу
        train_intercept = model.intercept_
        selected_features = ', '.join(features)
        st.write(f"Выбранные параметры: {selected_features}")
        st.write("Веса модели (коэффициенты):")
        st.write(pd.DataFrame(train_coefs, index=features, columns=["Weights"]))
         
        train_pred = model.predict(train[features])
        train_mse = model.score(train[features], train[y_true])
        #st.write("Предсказания на обучающем наборе данных:")
        #st.write(train_pred)
        st.write("Среднеквадратичная ошибка на обучающем наборе данных:")
        st.write(train_mse)

        # Сравнение предсказанных значений с реальными на обучающем наборе данных
        train_comparison_df = pd.DataFrame({'Actual': train[y_true], 'Predicted': train_pred})
        st.write("Сравнение предсказанных и реальных значений на обучающем наборе данных:")
        st.write(train_comparison_df)
            

        test_pred = model.predict(test[features])
        test_mse = model.score(test[features], test[y_true])
       # st.write("Предсказания на тестовом наборе данных:")
        #st.write(test_pred)
        st.write("Среднеквадратичная ошибка на тестовом наборе данных:")
        st.write(test_mse)

        test_comparison_df = pd.DataFrame({'Actual': test[y_true], 'Predicted': test_pred})
        st.write("Сравнение предсказанных и реальных значений на тестовом наборе данных:")
        st.write(test_comparison_df)

        # Визуализация результата на тестовом и обучающем наборах данных
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')# Добавление точек обучающего набора данных
        ax.scatter(train[features[0]], train[features[1]], train[y_true], c='r', marker='o', label='Train')# Добавление точек тестового набора данных
        ax.scatter(test[features[0]], test[features[1]], test[y_true], c='b', marker='o', label='Test')# Создание сетки для предсказаний модели
        x_range = np.linspace(min(train[features[0]].min(), test[features[0]].min()),
                      max(train[features[0]].max(), test[features[0]].max()), 10)
        y_range = np.linspace(min(train[features[1]].min(), test[features[1]].min()),
                      max(train[features[1]].max(), test[features[1]].max()), 10)
        X_range, Y_range = np.meshgrid(x_range, y_range)
        Z_range = model.predict(np.c_[X_range.ravel(), Y_range.ravel()]).reshape(X_range.shape)# Добавление плоскости предсказаний модели
        ax.plot_surface(X_range, Y_range, Z_range, alpha=0.5, color='g', label='Prediction Surface')
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(y_true)
        ax.set_title('3D Visualization')
        ax.legend()# Отображение графика
        st.pyplot(fig)
            
else:
    st.write("Пожалуйста, загрузите данные")




