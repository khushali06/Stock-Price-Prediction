# Stock Price Prediction (NIFTY 50)

🔗 **Kaggle Dataset Link:** [NIFTY 50 25 Yrs Data](https://www.kaggle.com/datasets/ashishjangra27/nifty-50-25-yrs-data)

This project focuses on predicting **NIFTY 50 stock prices** using a hybrid pipeline combining both **Machine Learning (ML)** and **Deep Learning (DL)** approaches. The workflow involves preparing time-series data, training multiple regression and neural network models, and rigorously evaluating and comparing their performance[cite: 1].

---

## 🛠️ Pipeline Overview

1. **Data Loading**
   - Loads historical stock price data (`data.csv`)[cite: 1].
   - Core features tracked: `Open`, `Close`, `High`, `Low`[cite: 1].

2. **Data Preparation**
   - Generates supervised learning datasets via sliding windows across multiple time horizons (**30 to 250 days**)[cite: 1].
   - Creates distinct `(X, y)` feature-target pairs for each time chunk[cite: 1].

3. **Model Architecture**
   - **Machine Learning Models**[cite: 1]  
     - *Linear:* `LinearRegression`, `Ridge`, `Lasso`[cite: 1]  
     - *Tree-Based & Ensembles:* `RandomForest`, `GradientBoosting`, `XGBoost`, `LightGBM`[cite: 1]  
     - *Others:* `SVR`, `KNeighborsRegressor`[cite: 1]
   - **Deep Learning Models**[cite: 1]  
     - Sequential neural network architectures built using Keras: `SimpleRNN`, `LSTM`, `GRU`, and `Bidirectional LSTM`[cite: 1].

4. **Training & Evaluation**
   - Trains models on rolling window subsets[cite: 1].
   - Tracks both training and testing errors using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** to measure generalization[cite: 1].

5. **Comparison**
   - Compares classical ML vs. Deep Learning performance across varying input window sizes to identify optimal configurations[cite: 1].

---

## ✨ Key Highlights

- **Hybrid Approach:** Evaluates both classical machine learning algorithms and sequential deep learning neural networks within a unified benchmarking framework[cite: 1].
- **Multi-Horizon Analysis:** Explores sliding window sizes ranging from 30 up to 250 days to test prediction robustness over short and long horizons[cite: 1].
- **Comprehensive Tracking:** Automated evaluation pipeline logging train/test MAE and RMSE metrics for dozens of model-window combinations[cite: 1].

---

## 🚀 Getting Started

### Prerequisites
Make sure you have the following libraries installed:
```bash
pip install numpy pandas scikit-learn xgboost lightgbm tensorflow tqdm matplotlib
