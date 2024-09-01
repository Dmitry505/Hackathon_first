import streamlit as st
import librosa
import numpy as np
import pickle

from Preprocessing import preprocessing

st.title("Предсказание аномалий")

# Загрузка аудиофайла
uploaded_file = st.file_uploader("Загрузите аудиофайл (WAV)", type=["wav"])

if uploaded_file is not None:
    signal, sr = librosa.load(uploaded_file)

    # Вывод исходного аудиофайла
    st.audio(signal, sample_rate=sr)

    st.markdown("""---""")

machine_type = st.radio(
    "выберите тип оборудования",
    ["pump", "fan", "slider", "valve"],
)

st.title("Получить предсказания на основе данных")


pump_forest = pickle.load(open(r'models/pump_forest.pkl', 'rb'))
slider_forest = pickle.load(open(r'models/slider_forest.pkl', 'rb'))
valve_forest = pickle.load(open(r'models/valve_forest.pkl', 'rb'))
fan_forest = pickle.load(open(r'models/fan_forest.pkl', 'rb'))


button_clicked = st.button("Предсказать")
if button_clicked:
    if machine_type == "valve":
        data = preprocessing(signal, sr, "valve")

        valve_forest_pred = valve_forest.predict(data)[0]
        st.write(f"Forest: {'Все в норме' if valve_forest_pred else 'Поломка'}")

    if machine_type == "pump":
        data = preprocessing(signal, sr, "pump")

        pump_forest_pred = pump_forest.predict(data)[0]
        st.write(f"Forest: {'Все в норме' if pump_forest_pred else 'Поломка'}")

    if machine_type == "slider":
        data = preprocessing(signal, sr, "slider")

        slider_forest_pred = slider_forest.predict(data)[0]
        st.write(f"Forest: {'Все в норме' if slider_forest_pred else 'Поломка'}")

    if machine_type == "fan":
        data = preprocessing(signal, sr, "fan")

        fan_forest_pred = fan_forest.predict(data)[0]
        st.write(f"Forest: {'Все в норме' if fan_forest_pred else 'Поломка'}")
