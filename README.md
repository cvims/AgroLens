# AgroLens

## Project Overview
This project was developed in collaboration with **Professor Dr. Torsten Schön** from **Technische Hochschule Ingolstadt** and **MI4People gGmbH**. The aim was to find a reliable estimate of soil quality samples using Sentinel-2 satellite data as a free alternative to laboratory tests. Multiple machine learning models were developed and evaluated to achieve this goal.

More details: [MI4People - Soil Quality Evaluation System](https://www.mi4people.org/soil-quality-evaluation-system)


## Purpose
Efficient nutrient management and precise fertilization are essential for advancing modern agriculture, particularly in regions striving to optimize crop yields sustainably. The AgroLens project endeavors to address this challenge by developing Machine Learning (ML)-based methodologies to predict soil nutrient levels without reliance on laboratory tests. By leveraging state of the art techniques, the project lays a foundation for actionable insights to improve agricultural productivity in resource-constrained areas, such as Africa.

The approach begins with the development of a robust European model using the LUCAS Soil dataset and Sentinel-2 satellite imagery to estimate key soil properties, including phosphorus, potassium, nitrogen, and pH levels. This model is then enhanced by integrating supplementary features, such as weather data, harvest rates, and Clay AI-generated embeddings.

This repository contains the code to the data preprocessing strategies and ML pipelines employed in this project. Advanced algorithms, including Random Forests, Extreme Gradient Boosting (XGBoost), and Fully Connected Neural Networks (FCNN), were implemented and fine-tuned for precise nutrient prediction. Results showcase robust model performance, with root mean square error values meeting stringent accuracy thresholds.


## Repository structure
- **nutrients_predictor/**: Implementation of the ML models developed during the project
- **satellite_utils/**: Modules responsible for querying and preprocessing satellite data
- **scripts/**: A collection of Python scripts used throughout the project, including data acquisition, preprocessing, and model evaluation tasks

The machine learning models can be executed using `nutrients_predictor/predictor_training.py`.


## Configuration
The following global environment variables are available:

### Paths
- **DATASET_PATH**: Directory containing the preprocessed input data CSV tables
- **MODEL_PATH**: Directory to save and load the trained ML models
- **SENTINEL_DIR**: Directory containing preprocessed Sentinel-2 data
- **LANDSAT_DIR**: Directory containing preprocessed Landsat 8 data
- **TMP_DIR**: Temporary directory used during satellite data preprocessing

### Credentials
- **COPERNICUS_USER**: Copernicus Data Space username (Sentinel-2 and Landsat 8 data)
- **COPERNICUS_PASSWORD**: Copernicus Data Space password or token (Sentinel-2 and Landsat 8 data)
- **AWS_ACCESS_KEY_ID**: Copernicus Data Space AWS key (Sentinel-2 cloud detection)
- **AWS_SECRET_ACCESS_KEY**: Copernicus Data Space AWS secret (Sentinel-2 cloud detection)
- **M2M_USER**: USGS M2M API username (Landsat 7 data)
- **M2M_SECRET**: USGS M2M API secret (Landsat 7 data)
- **OPENWEATHERMAP_KEY**: Open Weather Map API key (weather data)

## Project Members
- **Calvin Kammerlander**
- **Viola Kolb**
- **Marinus Luegmair**
- **Lou Scheermann**
- **Maximilian Schmailzl**
- **Marco Seufert**
- **Jiayun Zhang**


## Acknowledgments
We would like to express our gratitude to:
- **Prof. Dr. Torsten Schön** from **Technische Hochschule Ingolstadt**
 for supervising the project and providing valuable ML expertise.
- **Dr. Denis Dalic** from **MI4People** for establishing the project idea and delivering the initial information.
- **Prof. Dr. Patrick Noack** from **Hochschule Weihenstephan-Triesdorf** for his expert knowledge
 regarding fertilization and nutritional values.
