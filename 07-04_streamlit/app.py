import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, r2_score
from prophet import Prophet
import numpy as np

st.title("Временные ряды")
st.write("Файл CSV должен содержать данные по годам. Разбивка на train и test выборки осуществляется по выбранному году)")
uploaded = st.file_uploader("Загрузите файл CSV", type=["csv"])
if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.table(data.head(4))
    st.write("Пожалуйста, прежде чем двигаться дальше, задайте значения ниже")
    time = st.selectbox("Задайте столбец с периодом дат", data.columns)
    target = st.selectbox("Задайте параметр для прогноза", data.columns)
    data[time] = pd.to_datetime(data[time])
    data = data.set_index(time)


    if st.button("Показать график"):
        fig = plt.figure(figsize=(14, 6))
        plt.plot(data[target], marker='o')
        plt.xlabel('Time period')
        st.pyplot(fig)

    if st.button("Показать декомпозицию на тренд, сезонность и остатки"):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
        res = seasonal_decompose(data[target])
        res.observed.plot(ax=ax1)
        ax1.yaxis.set_label_position("right")
        ax1.set_ylabel("Observed", rotation=0, labelpad=20)
        res.trend.plot(ax=ax2)
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel("Trend", rotation=0, labelpad=20)
        res.seasonal.plot(ax=ax3)
        ax3.yaxis.set_label_position("right")
        ax3.set_ylabel("Seasonal", rotation=0, labelpad=20)
        res.resid.plot(ax=ax4)
        ax4.yaxis.set_label_position("right")
        ax4.set_ylabel("Residual", rotation=0, labelpad=20)
        st.pyplot(fig)  

    if st.button("Показать графики ACF и PACF"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))
        n_lags = 12
        acf = plot_acf(data[target], ax=ax1, lags=n_lags)
        pacf = plot_pacf(data[target], ax=ax2, lags=n_lags)
        st.pyplot(fig) 

    st.subheader("Prophet model")
    st.write("")
    data_prophet = data[target].reset_index().rename(columns={time: 'ds', target: 'y'}) 
    train_year = st.number_input("Выберите год окончания обучающей выборки", value=2010)
    test_year = st.number_input("Выберите год начала тестовой выборки", value=2017)
    seasonality_period = st.number_input("Выберите сезонность", value=52)

    data_train_prophet = data_prophet[data_prophet['ds'].dt.year >= train_year]
    data_test_prophet = data_prophet[data_prophet['ds'].dt.year >= test_year]

    model = Prophet()
    model.fit(data_train_prophet)

   # Создание фрейма данных для прогнозирования на data_test_prophet и еще на 52 недели вперед
    future = model.make_future_dataframe(periods=len(data_test_prophet) + seasonality_period, freq='W')
    forecast = model.predict(future)

    forecast_train = forecast[:-len(data_test_prophet) - seasonality_period] # Тренировочный период
    forecast_test = forecast[-len(data_test_prophet) - seasonality_period: -seasonality_period] # Тестовый период
    forecast_future = forecast[-seasonality_period:] # Будущий период
    prophet_mae_train = np.round(mean_absolute_error(data_train_prophet['y'], forecast_train['yhat']), 1)
    prophet_mae_test = np.round(mean_absolute_error(data_test_prophet['y'], forecast_test['yhat']), 1)

    st.write(f'mae train= {prophet_mae_train}')
    st.write(f'mae test = {prophet_mae_test}')

    # Визуализация прогнозов
    fig = plt.figure(figsize=(14, 6))
    plt.plot(data_prophet['ds'], data_prophet['y'], label='True data', marker='o')
    plt.plot(data_test_prophet['ds'], data_test_prophet['y'], label='Test data', marker='o')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', marker='o')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Prophet Forecast')
    plt.legend()
    st.pyplot(fig)  
            
else:
    st.write("Пожалуйста, загрузите данные")

