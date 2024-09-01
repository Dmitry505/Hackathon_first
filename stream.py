import streamlit as st
import librosa
import numpy as np
import pickle

def get_params(audio, sr):
	return abs(np.mean(librosa.feature.melspectrogram(y=audio, sr=sr),   axis=1))

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
    ["fan", "pump", "slider","valve"],
)


st.title("Получить предсказания на основе данных")

fan_Bagging = pickle.load(open(r'models/fan_Bagging.pkl', 'rb'))
fan_forest = pickle.load(open(r'models/fan_forest.pickle', 'rb')) 
pump_Bagging = pickle.load(open(r'models/pump_Bagging.pkl', 'rb'))
pump_forest = pickle.load(open(r'models/pump_forest.pickle', 'rb'))
#slider_Bagging = pickle.load(open(r'models/slider_Bagging.pickle', 'rb')) 
#slider_forest = pickle.load(open(r'models/slider_forest.pickle', 'rb'))
#valve_Bagging = pickle.load(open(r'models/valve_Bagging.pickle', 'rb')) 
#valve_forest = pickle.load(open(r'models/valve_forest.pickle', 'rb'))
    
button_clicked = st.button("Предсказать")
if button_clicked:
    data =  np.array(get_params(signal, sr)).reshape(1,-1)

    if machine_type == "fan":
        fan_Bagging_pred = fan_Bagging.predict(data)[0]
        st.write(f"Bagging: {fan_Bagging_pred}")

        fan_forest_pred = fan_forest.predict(data)[0]
        st.write(f"Forest: {fan_forest_pred}")
    
    if machine_type == "pump":
        pump_Bagging_pred = pump_Bagging.predict(data)[0]
        st.write(f"Bagging: {pump_Bagging_pred}")

        pump_forest_pred = pump_forest.predict(data)[0]
        st.write(f"Forest: {pump_forest_pred}")
    
    if machine_type == "slider":
        slider_Bagging_pred = slider_Bagging.predict(data)[0]
        st.write(f"Bagging: {slider_Bagging_pred}")

        slider_forest_pred = slider_forest.predict(data)[0]
        st.write(f"Forest: {slider_forest_pred}")
    
    if machine_type == "fan":
        valve_Bagging_pred = valve_Bagging.predict(data)[0]
        st.write(f"Bagging: {valve_Bagging_pred}")

        valve_forest_pred = valve_forest.predict(data)[0]
        st.write(f"Forest: {valve_forest_pred}")