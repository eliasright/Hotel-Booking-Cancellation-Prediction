import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cancellation_odds_transformer import CancellationOddsTransformer

import sys
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Hotel Reservations.csv'
df = pd.read_csv(file_path)
plt.figure(figsize=(10, 6))
df['arrival_month'].hist(bins=len(df['arrival_month'].unique()), edgecolor='black')
plt.title('Histogram of Arrival Months')
plt.xlabel('Month')
plt.ylabel('Frequency')
plt.grid(axis='y')

# Preparing the data
X = df.drop('booking_status', axis=1)
y = df['booking_status'].replace({'Canceled': 1, 'Not_Canceled': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define continuous and categorical columns (Without 'lead_time' as it will be used to calculate odds)
cols_continuous = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
                    'arrival_month', 'arrival_date', 
                   'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 
                   'avg_price_per_room', 'no_of_special_requests'] 
cols_categorical = ['type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 
                    'market_segment_type', 'repeated_guest']
cols_categorical = ['type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 
                    'market_segment_type', 'repeated_guest']

# Create transformers for the continuous and categorical columns
continuous_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

# Instantiate the custom transformer
cancellation_odds_transformer = CancellationOddsTransformer()

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, cols_continuous),
        ('cat', categorical_transformer, cols_categorical),
        ('custom', cancellation_odds_transformer, ['lead_time'])
    ])

# Create the pipeline with SMOTE and XGBoost Classifier
pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                              ('smote', SMOTE(random_state=42)),
                              ('classifier', XGBClassifier(random_state=100))])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the performance metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
