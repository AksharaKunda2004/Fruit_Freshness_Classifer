# Fruit Freshness Classifier - Deep Learning & MLOps

## Overview
This project is an end-to-end Deep Learning and MLOps application that classifies fruits as **Fresh** or **Stale** using the MobileNetV2 architecture. It demonstrates the complete machine learning lifecycle, including data preprocessing, model training, experiment tracking, containerization, CI/CD automation, and deployment.

---

## Features
- Fruit Freshness Classification (Fresh vs Stale)
- Transfer Learning using MobileNetV2
- Data Preprocessing & Augmentation
- MLflow Experiment Tracking
- Docker Containerization
- GitHub Actions CI/CD
- Streamlit Web Application
- Modular Project Structure

---

## Tech Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras, MobileNetV2
- **MLOps:** MLflow
- **Deployment:** Streamlit
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Version Control:** Git & GitHub

---

## Project Structure

```text
MLOP_FINAL_PROJECT/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── scripts/
├── dags/
├── model/
├── data/
└── README.md
```

---

## Installation

```bash
git clone https://github.com/AksharaKunda2004/Fruit_Freshness_Classifier.git
cd Fruit_Freshness_Classifier

pip install -r requirements.txt

streamlit run app.py
```

---

## Workflow

1. Data Collection
2. Data Preprocessing
3. Model Training using MobileNetV2
4. Experiment Tracking with MLflow
5. Docker Containerization
6. CI/CD using GitHub Actions
7. Streamlit Deployment
