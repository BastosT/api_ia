# pour construire l'image docker 
docker build -t flask-prediction-app .

# Lancer le conteneur avec le port 5001 
docker run -p 5001:5001 flask-prediction-app
