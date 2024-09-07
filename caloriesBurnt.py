import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
import streamlit as st
from io import StringIO
import numpy as np

# Load datasets with correct paths on your local machine
exercise_df = pd.read_csv('C:/Users/reena/OneDrive/Desktop/archive/exercise.csv')
calories_df = pd.read_csv('C:/Users/reena/OneDrive/Desktop/archive/calories.csv')


# Merge the dataframes on 'User_ID'
df = pd.merge(exercise_df, calories_df, on='User_ID')

st.title("Exercise and Calories Data Analysis")

# Display the first few rows of the dataframe
st.header("First Few Rows of the Merged Dataframe")
st.dataframe(df.head())

# Check the shape of the data
st.header("Shape of the Merged Dataframe")
st.write(df.shape)

# Get basic information about the data
st.header("Basic Information about the Merged Dataframe")
buffer = StringIO()
df.info(buf=buffer)
info = buffer.getvalue()
st.text(info)

# Get summary statistics
st.header("Summary Statistics of the Merged Dataframe")
st.write(df.describe())

# Display the column names
st.header("Column Names of the Merged Dataframe")
st.write(df.columns.tolist())

# Plot the distribution of the 'Duration' column
st.header("Distribution of Exercise Duration")
plt.figure()
plt.hist(df['Duration'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Exercise Duration')
st.pyplot(plt)

# Plot the distribution of the 'Calories' column
st.header("Distribution of Calories Burned")
plt.figure(figsize=(10, 6))
plt.hist(df['Calories'], bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.title('Distribution of Calories Burned')
st.pyplot(plt)

# Identify and highlight potential outliers in 'Calories'
q1 = df['Calories'].quantile(0.25)
q3 = df['Calories'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['Calories'] < lower_bound) | (df['Calories'] > upper_bound)]

# Highlight outliers on the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Calories'], bins=30, edgecolor='k', alpha=0.7)
plt.scatter(outliers['Calories'], [0] * len(outliers), color='red', label='Outliers')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.title('Calories with Potential Outliers')
plt.legend()
st.pyplot(plt)

st.write(f"Number of outliers in 'Calories': {len(outliers)}")
st.write("Outliers:")
st.dataframe(outliers)

# Drop the outliers for linear regression
df_clean = df[(df['Calories'] >= lower_bound) & (df['Calories'] <= upper_bound)]

# Define the features and target variable for Linear Regression
X_clean = df_clean[['Duration']]
y_clean = df_clean['Calories']

# Split the dataset into training and testing sets for clean data
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model with clean data
model_clean = LinearRegression()
model_clean.fit(X_train_clean, y_train_clean)

# Make predictions and evaluate performance on clean data
y_pred_clean = model_clean.predict(X_test_clean)
mse_clean = mean_squared_error(y_test_clean, y_pred_clean)
rmse_clean = np.sqrt(mse_clean)
r2_clean = r2_score(y_test_clean, y_pred_clean)

# Plot the regression line for clean data
st.header("Linear Regression: Duration vs. Calories (Clean Data)")
plt.figure(figsize=(10, 6))
plt.scatter(X_test_clean, y_test_clean, color='blue', label='Actual')
plt.plot(X_test_clean, y_pred_clean, color='red', linewidth=2, label='Predicted')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories')
plt.title('Linear Regression: Duration vs. Calories (Clean Data)')
plt.legend()
st.pyplot(plt)

st.header("Linear Regression Performance (Clean Data)")
st.write(f"Mean Squared Error: {mse_clean:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse_clean:.2f}")
st.write(f"R^2 Score: {r2_clean:.2f}")

# Logistic Regression and Confusion Matrix
# Binarize the target variable 'Calories' by taking the median as a threshold
median_calories = df['Calories'].median()
df['Calories_Binary'] = np.where(df['Calories'] >= median_calories, 1, 0)

# Define the features and target variable for Logistic Regression
X_logreg = df[['Duration']]  # Using 'Duration' as the feature
y_logreg = df['Calories_Binary']  # Binarized 'Calories' as the target

# Split the dataset into training and testing sets for Logistic Regression
X_train_logreg, X_test_logreg, y_train_logreg, y_test_logreg = train_test_split(X_logreg, y_logreg, test_size=0.2, random_state=42)

# Create and fit the Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_logreg, y_train_logreg)

# Make predictions on the test set
y_pred_logreg = logreg_model.predict(X_test_logreg)

# Compute the confusion matrix and classification report
conf_matrix_logreg = confusion_matrix(y_test_logreg, y_pred_logreg)
class_report_logreg = classification_report(y_test_logreg, y_pred_logreg)

# Display Confusion Matrix and Classification Report in Streamlit
st.header("Logistic Regression: Duration vs. Calories (Binary)")

st.subheader("Confusion Matrix")
st.write(conf_matrix_logreg)

st.subheader("Classification Report")
st.text(class_report_logreg)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
plt.matshow(conf_matrix_logreg, cmap=plt.cm.Blues, fignum=1)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
for (i, j), val in np.ndenumerate(conf_matrix_logreg):
    plt.text(j, i, val, ha='center', va='center')
st.pyplot(plt)
