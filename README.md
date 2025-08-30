# Noise Pollution Prediction

## Overview

This project leverages machine learning to predict urban noise pollution levels in India using historical data collected from 2011 to 2018. By analyzing environmental factors and location-based information, it uncovers patterns and trends in noise levels across different regions and time periods. The model provides actionable insights that can help city planners, policymakers, and researchers design quieter, healthier urban environments. Additionally, the project enables visualization of results to support data-driven decision-making for public health and urban management.

## Features

- Predicts noise levels from environmental and location data
- Visualizes trends and model performance
- Simple training and prediction workflow

## Usage

1. Clone the repo:
    ```bash
    git clone https://github.com/mizaaaaa/NoisePollution_prediction.git
    cd NoisePollution_prediction
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Add your dataset to the `data/` folder.
4. Train the model:
    ```bash
    python src/train.py
    ```
5. Predict noise levels:
    ```bash
    python src/predict.py --input your_input.csv
    ```

## Tech Stack

Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib

## License

MIT
