# Comparative Implementation of Machine Learning Regression Algorithms

## House Price Prediction Using Six Regression Models

This project compares six machine learning regression algorithms for
predicting house values using the **California Housing dataset**
provided by Scikit-learn.

The same dataset, input features, target variable, and 80:20
training/testing split are used for all six models to support a fair
comparison.

## 1. Project Objectives

The objectives of this project are to:

-   Understand the working principles and mathematical foundations of
    six regression algorithms.
-   Implement each algorithm using Python machine learning libraries.
-   Apply appropriate data preprocessing and feature scaling.
-   Evaluate model performance using MAE, MSE, RMSE, and R².
-   Compare actual and predicted house values visually.
-   Identify the best-performing model based on the experimental
    results.

## 2. Dataset

**Dataset:** California Housing dataset\
**Source:** Scikit-learn\
**Dataset documentation:**
https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

The dataset is loaded directly through Scikit-learn. A separate manual
CSV download is not required. When the dataset is not already cached,
Scikit-learn retrieves it as part of the dataset-loading process, so an
internet connection may be required.

The dataset contains 20,640 observations and eight input features.

### Input features

  Feature        Description
  -------------- ------------------------------------------
  `MedInc`       Median income in the block group
  `HouseAge`     Median house age in the block group
  `AveRooms`     Average number of rooms per household
  `AveBedrms`    Average number of bedrooms per household
  `Population`   Block group population
  `AveOccup`     Average number of household members
  `Latitude`     Block group latitude
  `Longitude`    Block group longitude

### Target variable

`MedHouseVal` represents the median house value for the block group. The
target is expressed in units of **\$100,000**. For example, a target
value of 2.0 corresponds to \$200,000.

## 3. Algorithms Implemented

The project implements the following six regression algorithms:

1.  **Linear Regression** --- estimates a linear relationship between
    the input features and house values.
2.  **Decision Tree Regression** --- predicts values by dividing
    observations into feature-based decision regions.
3.  **Random Forest Regression** --- combines predictions from multiple
    decision trees.
4.  **Neural Network Regression** --- learns relationships through
    interconnected neurons and weighted layers.
5.  **XGBoost Regression** --- builds boosted decision trees
    sequentially to improve prediction performance.
6.  **K-Nearest Neighbors (KNN) Regression** --- predicts a value using
    the target values of nearby training observations.

## 4. Project Workflow

The project follows this workflow:

1.  Load the California Housing dataset.
2.  Inspect the dataset, feature columns, target variable, and missing
    values.
3.  Separate the input features (`X`) and target (`y`).
4.  Split the data into 80% training data and 20% testing data.
5.  Apply feature scaling to algorithms that benefit from standardised
    inputs.
6.  Train all six regression models.
7.  Generate predictions for the test data.
8.  Calculate the four evaluation metrics.
9.  Create actual-versus-predicted visualisations.
10. Compare model performance and identify the strongest model based on
    the measured results.

## 5. Train-Test Split and Preprocessing

The dataset is divided using `train_test_split` with:

-   Training proportion: 80% (16,512 observations)
-   Testing proportion: 20% (4,128 observations)
-   Random state: `42`

`StandardScaler` is applied within a Scikit-learn pipeline for:

-   Linear Regression
-   Neural Network Regression
-   KNN Regression

The scaler is fitted on the training data only and then applied to the
test data. This prevents information from the test set from leaking into
preprocessing.

Decision Tree, Random Forest, and XGBoost do not require standardisation
for this implementation.

## 6. Evaluation Metrics

Each model is evaluated using the same four metrics:

  -----------------------------------------------------------------------
  Metric                  Meaning                 Preferred result
  ----------------------- ----------------------- -----------------------
  MAE                     Mean Absolute Error;    Lower
                          average absolute        
                          difference between      
                          actual and predicted    
                          values                  

  MSE                     Mean Squared Error;     Lower
                          average squared         
                          prediction error        

  RMSE                    Root Mean Squared       Lower
                          Error; square root of   
                          MSE                     

  R² Score                Proportion of target    Higher
                          variance explained by   
                          the model relative to   
                          the test-set mean       
                          baseline                
  -----------------------------------------------------------------------

The target is measured in units of \$100,000. Therefore, MAE and RMSE
are also expressed in those units, while MSE is expressed in squared
target units.

## 7. Requirements

The project uses Python 3 and the following libraries:

-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   XGBoost

Install the dependencies with:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## 8. How to Run

The notebook can be run in Google Colab or a local Jupyter environment.

### Google Colab

1.  Open https://colab.research.google.com/.

2.  Create a new notebook or upload the project notebook.

3.  Install the required packages if they are not already available:

    ``` python
    !pip install scikit-learn xgboost pandas numpy matplotlib seaborn
    ```

4.  Run the import and dataset-loading cells.

5.  Run the preprocessing and train-test split cells.

6.  Define and train all six models.

7.  Run the evaluation and plotting cells.

8.  Review the resulting metrics table and visualisations.

### Dataset loading code

``` python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print("Features:", X.shape)
print("Target:", y.shape)
```

The expected shapes are:

``` text
Features: (20640, 8)
Target: (20640,)
```

## 9. Expected Outputs

After running the notebook, the project should produce:

-   Dataset inspection output.
-   Training and testing data dimensions.
-   Trained versions of all six regression models.
-   Predicted house values for the test set.
-   A comparative table containing MAE, MSE, RMSE, and R² Score.
-   Actual-versus-predicted scatter plots for all six algorithms.
-   Identification of the best-performing model based on the
    experimental results.

**Note:** Exact metric values depend on the code, model parameters,
library versions, and data split. The best-performing algorithm must be
identified from the results produced by the completed experiment. No
performance ranking is assumed in this README.

## 10. Handwritten Implementation

The accompanying handwritten component should explain the working
procedure and mathematical equations of each algorithm. It should also
include small, clearly identified numerical examples showing how
predictions are calculated.

The six algorithms to cover are:

-   Linear Regression
-   Decision Tree Regression
-   Random Forest Regression
-   Neural Network Regression
-   XGBoost Regression
-   K-Nearest Neighbors Regression

Where simplified numerical examples are used to demonstrate the
mathematics, they should be labelled as illustrative examples rather
than actual records from the California Housing dataset.

## 11. Model Comparison and Conclusion

Compare the six algorithms using all four evaluation metrics. Discuss
differences in prediction error, explanatory performance, computational
requirements, interpretability, and limitations.

The conclusion should name the model supported by the experimental
results and justify the selection using its measured MAE, MSE, RMSE, and
R² Score. Avoid claiming that a model is universally the best based on a
single dataset and train-test split.

## 12. References

-   Scikit-learn. California Housing dataset documentation:
    https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
-   Scikit-learn. User Guide:
    https://scikit-learn.org/stable/user_guide.html
-   XGBoost. Python Package Documentation:
    https://xgboost.readthedocs.io/

------------------------------------------------------------------------

**Project:** Comparative Implementation of Machine Learning Regression
Algorithms\
**Application:** California House Price Prediction\
**Implementation language:** Python
