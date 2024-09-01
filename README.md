﻿# white team
Репозиторий участников хакатона

Цель соревнования - разработка алгоритмов для детектирования аномалий в работе промышленной аппаратуры на основе анализа акустической информации. 

Нашей командой кейс был разбит на 2 части:
В первой части стояла задача обучение ml модели на предоставленных данных.
Второй же частью являлась разработка приложения на streamlit для возможности предсказания аномалий.

Первой задачей при работе со звуком являлось преобразовать его в читаемый для машины вид, для этого была использована рекомендуемая библиотека librosa.

В ходе работы было необходимо извлечь спектральные признаки. Было изучено множество различных методов, но лучшим для данной задачи оказался melspectrogram. Которая позволяла работать с данными без дополнительных обработок шума

В дальнейшей работе нашей задачей было обучение самих моделей, были протестированы несколько различных вариантов среди которых лучшие результаты показали модели randomforest и bagging_classification

В конечном итоге у нас вышло 4 модели для каждого из видов машин. Где точность определения по каждой из категорий составила более 95%


Расположение файлов
models содержатся все обученные модели для 4х машин, для каждой есть предсказание от randomforest и bagging_classification 
ML ноутбуки c обработкой данных и обучением моделей
main.py точка входа в программу на streamlit 
Preprocessing содержит функцию для аудиофайла звука полученного от пользователя используется в  main.py

запуск streamlit
streamlit run main.py
