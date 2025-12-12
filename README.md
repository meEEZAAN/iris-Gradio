# Iris Species Classification – Gradio + Docker

This project deploys a machine learning model that predicts the species of an Iris flower
(Setosa, Versicolor, or Virginica) based on four input features.

The model is served through a Gradio web interface and deployed inside a Docker container
for reproducibility and portability.

---

## Features
- Interactive Gradio UI with sliders
- Predicts Iris species from flower measurements
- Displays class probabilities
- Dockerized deployment

---

## Project Structure
```
text
iris-gradio/
├── app.py
├── model.joblib
├── requirements.txt
├── Dockerfile
└── README.md
```
## How to run with Docker
```
1. Build the Docker image
docker build -t iris-gradio-app .

2. Run the container
docker run -p 7860:7860 iris-gradio-app
```
## 3. Open the app
```
Open a browser and go to:

http://localhost:7860