# Sport_car_price_prediction

## Project Overview
This project focuses on predicting sports car prices using machine learning techniques. The workflow includes data exploration, preprocessing, feature engineering, model selection, and evaluation. The dataset contains information about various sports cars, including their make, model, year, engine specifications, performance metrics, and prices.

## Project Structure
The project consists of three main files:
1. `Project.ipynb` - Jupyter notebook containing the complete analysis and modeling pipeline
2. `Functions.py` - Python script with custom functions used in the notebook
3. `Sport_car_price.csv` - Dataset containing sports car information

## Requirements
To run this project, you'll need:
- Python 3.x
- Jupyter Notebook
- Required Python libraries (listed in the notebook)

## Running on Google Colab
To run this project on Google Colab:

1. Upload all three files to your Google Colab environment:
   - `Project.ipynb`
   - `Functions.py`
   - `Sport_car_price.csv`

2. Install the required libraries that aren't available by default in Colab. Most importantly, install CatBoost with:
   ```pip install catboost
   ```

3. Note that the University of Bologna logo image referenced in the notebook will not be visible in Colab as the file `unibo_logo.png` is not provided. This doesn't affect the functionality of the code.

## Key Steps in the Project
1. **Data Loading and Exploration**: Initial inspection of the dataset structure and contents
2. **Data Cleaning and Preprocessing**:
   - Handling missing values and placeholders
   - Correcting outliers and removing duplicates
   - Converting data types
3. **Feature Engineering**:
   - Creating new features like 'Engine Type'
   - Applying power transformations to skewed features
   - Encoding categorical variables
4. **Model Development**:
   - Comparing multiple regression models (including CatBoost, XGBoost, Random Forest, etc.)
   - Hyperparameter tuning for the best performing model
5. **Model Evaluation**:
   - Performance metrics (R², MSE, MAE)
   - Visualization of results and residuals

## Results
The best performing model was CatBoost, achieving:
- Validation R² score: 0.932
- Test R² score: 0.909 (after tuning)
- Significant improvement over linear regression (R²: 0.591)

## How to Use
1. Ensure all files are in the same directory
2. Run the Jupyter notebook cells sequentially
3. The notebook is self-contained with all necessary explanations

## Notes
- The project uses `random_state=42` for reproducibility
- Some visualizations may look different in Colab compared to local Jupyter environments
- The notebook includes detailed comments explaining each step

## Future Improvements
Suggestions for extending the project are included in the notebook's conclusion section.

## Contributors
- **Vittorio Caputo** 
- **Federico Collepardo**
