#  House Price Prediction using Linear Regression

This is a beginner-friendly machine learning project where we predict the sale prices of houses using basic features such as square footage (GrLivArea), number of bathrooms (FullBath), and number of bedrooms (BedroomAbvGr). This project uses the popular House Prices dataset from Kaggle.

## Dataset
- Source: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Files used:
    - `train.csv` placed inside the `data/` folder.

##  Technologies Used
- Python
- pandas
- scikit-learn
- matplotlib
- seaborn

##  Project Workflow
1. Data is loaded from `train.csv`.
2. Features selected:
    - GrLivArea (above ground living area in sq. ft.)
    - FullBath (number of full bathrooms)
    - BedroomAbvGr (number of bedrooms)
3. Linear Regression model is trained to predict `SalePrice`.
4. Model is evaluated using:
    - Mean Squared Error (MSE)
    - R² Score
5. Scatter plot is generated comparing Actual vs Predicted House Prices.
