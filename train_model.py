import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib  # Pour sauvegarder et charger le scaler

# Charger les données
df = pd.read_csv("data/HomeC.csv", usecols=['temperature', 'time'])

# Vérifier les données
print(df.head())

# Normaliser la colonne température
scaler = MinMaxScaler(feature_range=(0, 1))
temps_scaled = scaler.fit_transform(df['temperature'].values.reshape(-1, 1))

# Sauvegarder le scaler pour l'utiliser lors des prédictions
joblib.dump(scaler, 'scaler.pkl')

# Préparer les séquences pour LSTM
seq_length = 2
X, y = [], []

for i in range(len(temps_scaled) - seq_length):
    X.append(temps_scaled[i:i+seq_length])
    y.append(temps_scaled[i+seq_length])

X = np.array(X)
y = np.array(y)

# Définir le modèle LSTM
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner le modèle
model.fit(X, y, epochs=10, batch_size=32)

# Sauvegarder le modèle
model.save("model_temperature.h5")
print("Modèle entraîné et sauvegardé sous 'model_temperature.h5'")
