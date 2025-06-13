

![CI](https://github.com/AnnaGanisheva/sales-predictor/actions/workflows/ci.yml/badge.svg)


Before start:
install required packages
pip install -r requirements.txt

# 🧠 Rossmann Sales Forecasting — End-to-End MLOps Project

This repository contains a pet project designed to demonstrate a **complete machine learning lifecycle** using **MLOps best practices**. The goal is to forecast weekly sales for Rossmann stores based on historical data, using a reproducible, scalable, and maintainable ML pipeline.

---

## 📊 Project Overview

- **Dataset:** Historical sales data from Rossmann drugstores.
- **Prediction target:** Weekly total sales for each store.
- **Prediction horizon:** 7 days ahead.

TODO
кращий вибір: XGBoost або LightGBM.
- **Model:** Tree-based regressors (e.g., Random Forest) with feature engineering.

---

## 🛠️ Data Transformation

In this step, the preprocessed raw data is loaded and transformed to be ready for training. The following actions are applied:

- Columns with missing values are dropped
- The `StateHoliday` column is cast to string type to avoid dtype issues
- Categorical variables (`StateHoliday`, `StoreType`, `Assortment`) are cast to `category` dtype for compatibility with LightGBM/XGBoost
- The `Date` column is excluded from categorical processing

---

## 🧠 Model Training

Two machine learning models were trained and evaluated: **XGBoost** and **LightGBM**.  
Hyperparameter optimization was performed using [Optuna](https://optuna.org/) with **Root Mean Squared Error (RMSE)** as the objective metric.

### 🧪 Training Procedure

- Preprocessed data was used as input
- Data was split into training and validation sets (80/20)
- Each model was tuned using Optuna (`n_trials=30`)
- Metrics and parameters were tracked and logged with **MLflow**
- Best-performing models were registered in the **MLflow Model Registry**

### 📊 Results
TODO: check error values

| Model     | RMSE       | MAPE       |
|-----------|------------|------------|
| XGBoost   | > 891.79   | > 10.54 %  |
| LightGBM  | **~891.79** | **~10.54%** |

> ⚡️ **LightGBM** achieved the best results and was selected as the final model for deployment.

The final LightGBM model was logged to MLflow and registered under the name:
``lightgbm_sales_forecaster``.

---

## 🎯 Project Objectives

- Practice structuring an ML project for production use
- Integrate key MLOps tools and workflows
- Enable reproducible training, testing, and deployment
- Visualize predictions through an interactive UI

---

## 🧰 Tech Stack

| Purpose             | Tool/Library                         |
|---------------------|--------------------------------------|
| Experiment tracking | **MLflow**                           |
| Pipeline orchestration | **Apache Airflow**               |
| Versioning / CLI    | DVC, Makefile, YAML config structure |
| Containerization    | **Docker**                           |
| CI/CD               | **GitHub Actions**                   |
| Monitoring          | **Grafana + Prometheus**             |
| UI                  | **Streamlit**                        |
| Testing / Quality   | Pytest, Coverage, Black, Flake8      |

---

## 🔄 Pipeline Execution

The Airflow pipeline is scheduled to run **once a week**, shortly before the beginning of the new sales week. This ensures that the model always uses the latest data to forecast the upcoming week's sales.

Typical pipeline flow:

Load → Clean → Feature Engineering → Train Model → Evaluate → Register in MLflow


---

## 📦 Dockerized Architecture

The entire system runs inside **Docker containers**, including:

- ML pipeline
- Airflow scheduler
- MLflow tracking server
- UI application (Streamlit)
- Monitoring stack (Grafana + Prometheus)

All services are orchestrated with `docker-compose`.

---

## ✅ Key Features

- ✅ Modular and testable ML codebase
- ✅ MLflow integration with parameter tracking and model registry
- ✅ Scheduled retraining via Airflow DAG
- ✅ Interactive UI for prediction testing
- ✅ Grafana dashboard for monitoring model quality and freshness
- ✅ CI/CD workflow using GitHub Actions

---

## 🚀 Live Demo (optional)

> _(Add here a link to your deployed Streamlit app or a GIF demo if available)_
TODO

---

## 📁 Project Structure

TODO: add final project structure


---

## 📍 Notes

This project is not intended for production use, but rather as an **educational showcase** of modern MLOps practices — combining engineering, automation, and ML experimentation in a clean, structured workflow.

