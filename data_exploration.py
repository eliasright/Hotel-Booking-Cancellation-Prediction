import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Hotel Reservations.csv'
df = pd.read_csv(file_path)
df['booking_status'] = np.where(df['booking_status'] == 'Not_Canceled', 0, 1)
df['lead_time_quartiles'] = pd.qcut(df['lead_time'], 4, labels=False)

# Check for null values
null_values = df.isnull().sum()
print(f"Null Values:\n{null_values}")

# Check for duplicates
duplicates = df.duplicated().sum()
print("\nNumber of Duplicate Rows:", duplicates)

# Split dataset
col_target = 'booking_status'
df_target = df[col_target]

cols_continuous = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
                'lead_time', 'arrival_year', 'arrival_month', 'arrival_date', 
                'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 
                'avg_price_per_room', 'no_of_special_requests']
df_continuous = df[cols_continuous] # Note: Discrete features like no_of_adults will be considered as continuous

cols_categorical = ['type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 
            'market_segment_type', 'repeated_guest', 'lead_time_quartiles']
df_categorical = df[cols_categorical]

# Continuous features
summary_statistics = df_continuous.describe()
print("\nSummary Statistics")
print(summary_statistics.transpose())

# Categorical features
print("\nCategorical Features")
for col in cols_categorical + [col_target]:
    unique_values = df[col].unique()
    print(f"Unique Values '{col}' (total {len(unique_values)}): {unique_values}")

# Boxplot for continuous outliers
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for i, ax in enumerate(axes.flatten()):
    sns.boxplot(x=df_continuous[cols_continuous[i]], ax=ax)
    ax.set_title(cols_continuous[i], fontsize=10)
    ax.set_xlabel('')
fig.subplots_adjust(hspace=0.3, wspace=0.1)  
plt.show()

# Pearson correlation
pearson_corr = pd.concat([df_continuous, df_target], axis=1).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Matrix')
plt.subplots_adjust(bottom=0.4, left=0.3)
plt.show()

# Chi-squared test
df_chi = df[cols_categorical + [col_target]].astype('category')
chi_squared_p_values = pd.DataFrame(index=cols_categorical, columns=[col_target], dtype=float)
for col in cols_categorical:
    confusion_matrix = pd.crosstab(df_chi[col], df_chi[col_target])
    # Check for expected frequencies condition
    if not np.all(confusion_matrix.values.flatten() >= 5):
        print(f"\nFailed to meet min 5 rule assumption: {col}")
        print(confusion_matrix.values)
    chi2, p, _, _ = chi2_contingency(confusion_matrix)
    chi_squared_p_values.loc[col, col_target] = p

plt.figure(figsize=(8, 6))
sns.heatmap(chi_squared_p_values, annot=True, cmap='coolwarm_r', fmt='.2e')
plt.title("Chi-squared against Booking Status")
plt.subplots_adjust(left=0.3)
plt.show()