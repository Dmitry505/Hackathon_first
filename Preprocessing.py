import librosa
import numpy as np
import pandas as pd


types = {
	'valve': [60, 61, 62, 82, 83, 84, 85, 99, 100, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 122, 123, 124, 127],
	'fan': [0, 1, 2, 3, 4, 8, 9, 10, 12, 13, 14, 15, 19, 24, 27, 35, 36, 48, 54, 70, 73, 76, 80, 81],
	'pump': [13, 16, 17, 18, 19, 20, 21, 50, 59, 60, 68, 69, 79, 80, 82, 83, 84, 85, 91, 92, 93, 94, 95, 96],
	'slider': [21, 75, 77, 78, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 113, 118, 121, 122, 124, 125, 126, 127],
}


def preprocessing(signal, sr, type):
	"""
	:param signal: аудиозапись
	:param sr: кол - во сэмплов
	:param type: тип машины
	:return: датафрейм с признаками
	"""
	X = pd.DataFrame(abs(np.mean(librosa.feature.melspectrogram(y=signal, sr=sr), axis=1))).T
	X_new = X[types[type]]
	return X_new
