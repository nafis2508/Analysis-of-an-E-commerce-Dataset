# Analysis of an E-commerce Dataset Part 2

## Overview

This project involves training linear regression models to predict user ratings for items in an e-commerce dataset. The analysis includes data exploration, feature selection, model training, and evaluation. We aim to investigate the impact of different feature selections and training/testing data sizes on model performance.

## Dataset

The dataset used in this analysis is a cleaned e-commerce dataset provided in the file `cleaned_ecommerce_dataset.csv`. It contains 2,685 entries and 11 columns.

## Project Structure

- **data**: Contains the dataset file.
- **notebooks**: Jupyter notebooks used for data analysis and modeling.
- **scripts**: Python scripts for data processing and model training.
- **README.md**: Project documentation.

## Installation

To run the analysis, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ecommerce-analysis-part2.git
   cd ecommerce-analysis-part2
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install and required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage 

1. **Load the Dataset**: Use the read_csv method from Pandas to load the dataset.
   ```bash
   import pandas as pd
   df = pd.read_csv('data/cleaned_ecommerce_dataset.csv')
2. **Explore the Dataset**:
   ```bash
   print(df.head())
   df.info()
3. **Feature Selection and Correlation Analysis:
   ```bash
   from sklearn.preprocessing import OrdinalEncoder

   encoder = OrdinalEncoder()
   df[["gender","category", "review"]] = encoder.fit_transform(df[["gender","category", "review"]])
   correlations = df[['helpfulness', 'gender', 'category', 'review', 'rating']].corr()
   print(correlations)
4. **Split Training and Testing Data**:
   ```bash
   from sklearn.model_selection import train_test_split

   X = df.drop(columns=['rating'])
   y = df['rating']

   X_train_case1, X_test_case1, y_train_case1, y_test_case1 = train_test_split(X, y, test_size=0.9, random_state=42)
   X_train_case2, X_test_case2, y_train_case2, y_test_case2 = train_test_split(X, y, test_size=0.1, random_state=42)
5. **Train Linear Regression Models**:
   ```bash
   from sklearn.linear_model import LinearRegression

   selected_features_most_correlated = ['category', 'gender']
   selected_features_least_correlated = ['helpfulness', 'review']

   model_a = LinearRegression().fit(X_train_case1[selected_features_most_correlated], y_train_case1)
   model_b = LinearRegression().fit(X_train_case1[selected_features_least_correlated], y_train_case1)
   model_c = LinearRegression().fit(X_train_case2[selected_features_most_correlated], y_train_case2)
   model_d = LinearRegression().fit(X_train_case2[selected_features_least_correlated], y_train_case2)
6. **Evaluate Models**:
   ```bash
   from sklearn.metrics import mean_squared_error
   import numpy as np
    
   def calculate_rmse(mse):
      return np.sqrt(mse)
    
    predictions_a = model_a.predict(X_test_case1[selected_features_most_correlated])
    mse_a = mean_squared_error(y_test_case1, predictions_a)
    rmse_a = calculate_rmse(mse_a)
    
    predictions_b = model_b.predict(X_test_case1[selected_features_least_correlated])
    mse_b = mean_squared_error(y_test_case1, predictions_b)
    rmse_b = calculate_rmse(mse_b)
    
    predictions_c = model_c.predict(X_test_case2[selected_features_most_correlated])
    mse_c = mean_squared_error(y_test_case2, predictions_c)
    rmse_c = calculate_rmse(mse_c)
    
    predictions_d = model_d.predict(X_test_case2[selected_features_least_correlated])
    mse_d = mean_squared_error(y_test_case2, predictions_d)
    rmse_d = calculate_rmse(mse_d)
    
    print(f'Model A MSE: {mse_a}, RMSE: {rmse_a}')
    print(f'Model B MSE: {mse_b}, RMSE: {rmse_b}')
    print(f'Model C MSE: {mse_c}, RMSE: {rmse_c}')
    print(f'Model D MSE: {mse_d}, RMSE: {rmse_d}')
7. **Visualize Results**:
   ```bash
   import matplotlib.pyplot as plt

    mse_results = [mse_a, mse_b, mse_c, mse_d]
    rmse_results = [rmse_a, rmse_b, rmse_c, rmse_d]
    models = ['Model A', 'Model B', 'Model C', 'Model D']
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(models, mse_results, color='skyblue')
    plt.title('Mean Squared Error (MSE)')
    plt.ylabel('MSE')
    plt.grid(axis='y')
    
    plt.subplot(1, 2, 2)
    plt.bar(models, rmse_results, color='lightgreen')
    plt.title('Root Mean Squared Error (RMSE)')
    plt.ylabel('RMSE')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.show()

## Contributing 
   Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License
   This project is licensed under the MIT License.

## Authors
-  Muntasir Md Nafis(https://github.com/nafis2508)(www.linkedin.com/in/muntasir-md-nafis)

## Additional Information 

- https://ilearn.mq.edu.au/mod/page/view.php?id=7936865








