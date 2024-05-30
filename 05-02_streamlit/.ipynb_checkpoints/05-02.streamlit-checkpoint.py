import streamlit as st
from skimage import io
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt

st.title("Изменение размерности изображения с помощью CVD")
st.header('Вставьте url')
url = st.text_input('Введите URL изображения:')
if url:
    image = io.imread(url)[:, :, 0]    
    U, sing_values, V = np.linalg.svd(image)
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(sigma, sing_values)    
    top_k = st.slider('Выберите количество сингулярных чисел (k):', min_value=1, max_value=min(image.shape), value=20)    
    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :] 
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(U @ sigma @ V, cmap='gray')
    ax[0].set_title('Полное SVD')
    ax[1].imshow(trunc_U @ trunc_sigma @ trunc_V, cmap='gray')
    ax[1].set_title(f'Усеченное SVD с K={top_k}')
    st.pyplot(fig)
else:
    st.write('Пожалуйста, введите URL изображения.')

#fig, ax = plt.subplots(1, 2, figsize=(15, 10))

#ax[0].imshow(U@sigma@V, cmap='grey')
#ax[1].imshow(trunc_U@trunc_sigma@trunc_V, cmap='grey')
