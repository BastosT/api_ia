# IoT Temperature Predictor

## 📝 Description
API Flask pour la prédiction de température utilisant des modèles d'apprentissage automatique basés sur les données historiques des capteurs IoT.

## 🛠️ Technologies
- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy
- requests


1. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 🤖 Modèle d'IA Utilisé

### Régression Linéaire
- Utilise scikit-learn's `LinearRegression`
- Normalisation des données avec `MinMaxScaler`
- Prédictions basées sur les tendances temporelles
- Adaptée pour les prédictions à court terme

## 🚀 Démarrage

```bash
flask run --port 5001
```

## 📚 API Endpoints

### Prédiction de Température
```http
GET /predict_horizon?horizon=10
```

#### Paramètres
- `horizon`: Nombre de points de prédiction (défaut: 10)

#### Réponse
```json
{
    "current_temperature": 22.4,
    "last_measurement_time": 1736324821,
    "predictions": [
        {
            "time": 1736324881,
            "temperature": 22.5
        }
    ],
    "horizon_minutes": 10
}
```

## 📊 Pipeline de Prédiction

1. **Collecte des Données**
   - Récupération via l'API de collecte
   - Validation des données

2. **Prétraitement**
   - Normalisation (MinMaxScaler)
   - Structuration temporelle

3. **Prédiction**
   - Entraînement du modèle
   - Génération des prédictions
   - Dénormalisation des résultats

## 🔄 Utilisation avec Docker

1. Build de l'image
```bash
docker build -t flask-prediction-app .
```

2. Lancement du container
```bash
docker run -p 5001:5001 flask-prediction-app
```

