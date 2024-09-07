# Exercise and Calories Data Analysis

This Streamlit application performs data analysis on exercise and calorie datasets using both Linear Regression and Logistic Regression. The app allows users to visualize data distributions, detect outliers, and evaluate regression models.

## Features

- **Data Loading:** Load exercise and calorie datasets and merge them based on `User_ID`.
- **Data Exploration:** Display basic information, shape, and summary statistics of the merged dataframe.
- **Data Visualization:** 
  - Histogram of exercise duration.
  - Histogram of calories burned.
  - Outlier detection in calorie data with visualization.
- **Linear Regression Analysis:** 
  - Predict calories burned based on exercise duration.
  - Display regression line and performance metrics.
- **Logistic Regression Analysis:**
  - Predict if calories burned are above or below the median value.
  - Display confusion matrix and classification report.

## Requirements

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/exercise-calories-analysis.git
    ```

2. Navigate to the project directory:
    ```bash
    cd exercise-calories-analysis
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `requirements.txt` file with the following content:
    ```
    pandas
    matplotlib
    scikit-learn
    streamlit
    numpy
    ```

## Usage

1. Place your datasets (`exercise.csv` and `calories.csv`) in the `archive` folder inside the project directory.
2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3. Open the URL provided by Streamlit in your web browser to interact with the app.

## Data Format

- **`exercise.csv`**
  - `User_ID`: Unique identifier for users.
  - `Duration`: Exercise duration in minutes.

- **`calories.csv`**
  - `User_ID`: Unique identifier for users.
  - `Calories`: Calories burned.

## Example

**Data Visualization:**
- Histograms of exercise duration and calories burned.
- Highlighted outliers in calories burned.

**Linear Regression Analysis:**
- Scatter plot and regression line of exercise duration vs. calories.
- Performance metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R^2 Score.

**Logistic Regression Analysis:**
- Confusion Matrix and Classification Report for predicting if calories burned are above or below the median value.


