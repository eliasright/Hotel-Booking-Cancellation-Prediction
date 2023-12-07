# Hotel Booking Cancellation Prediction
## How to run

1. **Download the Repository**: Clone or download the repository from GitHub to your local machine.

2. **Install Requirements**: Run `pip install -r requirements.txt` to install the required Python packages listed in `requirements.txt`.

3. **Run the Main Script**: Execute the main script of the project by running `python main.py`.

## Motivation
The motivation behind this test XGBoost in predict whether a guest would cancel a booking. Then try to finetune the model to improve upon the metrics: Accuracy, Precision, Recall, and F1. The predictive model will be trained using a training set, finetune on a validation set (called X_test in the code), and testing set (called X_final in the code). This way, I can improve on the model and use the validation set to guide me in improving the model. Just to make sure I am not overfitting to the validation set, the final testing set will be only tested at the end to see how well the model performed.

The dataset is split into 3 type of features:
- Continuous features
- Categorical features
- 'lead_time' -> 'prior_probability'

Here 'lead_time' refers to the duration between the booking date and the date of arrival. A custom transformer will be utilized to categorize customers into different clusters based on their 'lead_time'. Subsequently, for each of these clusters, the likelihood of cancellation is calculated. For instance, if a cluster comprises 6 people and 4 out of these 6 individuals cancel while 2 do not, a prior probability of 4/6 is assigned to that cluster. This probability will then be applied to any new data that falls within the same 'lead_time' range for that particular cluster.

Initially, in Model 1, the 'lead_time' is divided into four quartiles, representing the lower 25%, 25-50%, 50-75%, and 75-100% ranges. However, to enhance the model's effectiveness, Bayesian optimization will be employed later to determine an optimal split. This approach may result in a different partitioning of the 'lead_time', such as into two or eight segments, instead of the initial four

There will be a total of 3 models:
- Baseline model: XGBoost with default hyperparameters
- Bayesian Optimized XGBoost
- Bayesian Optimized XGBoost with feature selections

**Dataset:** Hotel Reservations Dataset

**Link:** https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset/data

**NOTE:** Dataset is considered to be data collected over a year and a half.

## Project Overview

1. [Data Overview and Preprocessing](#1-data-overview-and-preprocessing)
- Dataset: 2240 instances, 29 features.
- Steps: Handling missing values, duplicates, outliers; categorical data encoding; dataset standardization.

2. [Customer Segmentation](#2-customer-segmentation)
- Method: K-Means clustering, supported by PCA.
- Result: Identification of three distinct customer groups.

3. [Customer Profiling](#3-customer-profiling)
- Analysis: Detailed profiling based on demographics, income, and spending patterns across customer groups.

4. [Predictive Model Analysis](#4-predictive-model-analysis-for-latest-campaign)
- Approach: XGBoost algorithm for campaign responsiveness.
- Outcome: Enhanced understanding of varied customer responses to marketing campaigns.

### Summary of Findings

- Identified three unique customer segments with distinct characteristics.
- Most decisive differences between customer were income, spending, and age.
- Recall is improved when predicting customer response to a campaign by separating into customer clusters. Though not by much.
- There could be a benefit from using KNN or a clustering method to first assign a customer into a cluster before predicting customer's response to a campaign.

## 1. Data Overview and Preprocessing

### A. The dataset contains 36275 instances 19 features

    Booking_ID: unique identifier of each booking
    no_of_adults: Number of adults
    no_of_children: Number of Children
    no_of_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
    no_of_week_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
    type_of_meal_plan: Type of meal plan booked by the customer:
    required_car_parking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
    room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
    lead_time: Number of days between the date of booking and the arrival date
    arrival_year: Year of arrival date
    arrival_month: Month of arrival date
    arrival_date: Date of the month
    market_segment_type: Market segment designation.
    repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
    no_of_previous_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking
    no_of_previous_bookings_not_canceled: Number of previous bookings not canceled by the customer prior to the current booking
    avg_price_per_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
    no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
    booking_status: Flag indicating if the booking was canceled or not.

### B. General data overview

Data has no duplication or null values. 

#### Quick Summary of the continuous values

| Feature                                | count   | mean         | std         | min   | 25%   | 50%   | 75%   | max   |
|----------------------------------------|---------|--------------|-------------|-------|-------|-------|-------|-------|
| no_of_adults                           | 36275.0 | 1.844962     | 0.518715    | 0.0   | 2.0   | 2.00  | 2.0   | 4.0   |
| no_of_children                         | 36275.0 | 0.105279     | 0.402648    | 0.0   | 0.0   | 0.00  | 0.0   | 10.0  |
| no_of_weekend_nights                   | 36275.0 | 0.810724     | 0.870644    | 0.0   | 0.0   | 1.00  | 2.0   | 7.0   |
| no_of_week_nights                      | 36275.0 | 2.204300     | 1.410905    | 0.0   | 1.0   | 2.00  | 3.0   | 17.0  |
| lead_time                              | 36275.0 | 85.232557    | 85.930817   | 0.0   | 17.0  | 57.00 | 126.0 | 443.0 |
| arrival_year                           | 36275.0 | 2017.820427  | 0.383836    | 2017  | 2018  | 2018  | 2018  | 2018  |
| arrival_month                          | 36275.0 | 7.423653     | 3.069894    | 1.0   | 5.0   | 8.00  | 10.0  | 12.0  |
| arrival_date                           | 36275.0 | 15.596995    | 8.740447    | 1.0   | 8.0   | 16.00 | 23.0  | 31.0  |
| no_of_previous_cancellations           | 36275.0 | 0.023349     | 0.368331    | 0.0   | 0.0   | 0.00  | 0.0   | 13.0  |
| no_of_previous_bookings_not_canceled   | 36275.0 | 0.153411     | 1.754171    | 0.0   | 0.0   | 0.00  | 0.0   | 58.0  |
| avg_price_per_room                     | 36275.0 | 103.423539   | 35.089424   | 0.0   | 80.3  | 99.45 | 120.0 | 540.0 |
| no_of_special_requests                 | 36275.0 | 0.619655     | 0.786236    | 0.0   | 0.0   | 0.00  | 1.0   | 5.0   |

#### Categorical Features 

- **Unique Values 'type_of_meal_plan' (total 4):** 
  - 'Meal Plan 1'
  - 'Not Selected'
  - 'Meal Plan 2'
  - 'Meal Plan 3'

- **Unique Values 'required_car_parking_space' (total 2):** 
  - 0 
  - 1

- **Unique Values 'room_type_reserved' (total 7):** 
  - 'Room_Type 1'
  - 'Room_Type 4'
  - 'Room_Type 2'
  - 'Room_Type 6'
  - 'Room_Type 5'
  - 'Room_Type 7'
  - 'Room_Type 3'

- **Unique Values 'market_segment_type' (total 5):** 
  - 'Offline'
  - 'Online'
  - 'Corporate'
  - 'Aviation'
  - 'Complementary'

- **Unique Values 'repeated_guest' (total 2):** 
  - 0
  - 1

- **Unique Values 'lead_time_quartiles' (total 4):** 
  - 3
  - 0
  - 1
  - 2

- **Unique Values 'booking_status' (total 2):** 
  - 0
  - 1

Note: lead_time_quartiles is what was mentioned in [Motivation](#motivation)

### C. Continuous Features

Looking at the continuous features (including discrete values). First we look at the boxplots. 

Four features in your dataset: "no_of_adults", "no_of_children", "no_of_previous_cancellations", and "no_of_previous_bookings_not_cancelled". Each of these features represents a count and exhibits significant skewness, characterized by a dominant population with a single large number and numerous outliers. These outliers are not anomalies but rather true outliers that are expected in real-world scenarios. Retaining these outliers in the dataset is beneficial as they are representative of conditions you are likely to encounter in future data.

![Boxplot](https://github.com/eliasright/Hotel-Booking-Cancellation-Prediction/assets/151723828/fa4ba604-d38d-4945-968e-97756e98ded2)

Looking at the Pearson Correlation Matrix, we see that the majority of features have very weak or weak linear correlations. This implies that, for most pairs of features, changes in one variable do not reliably predict changes in the other. When building predictive models, a dataset like this could lead to overfitting if not careful. Since there are no strong linear relationships, the model might pick up on noise instead of actual patterns. There is a benefit in using Regularization like L1 and L2 in XGBoost to avoid this.

![Pearson-Correlation](https://github.com/eliasright/Hotel-Booking-Cancellation-Prediction/assets/151723828/119496dd-ddf4-410e-8b07-c8a757613302)

### D. Categorical Features

When performing a Chi-squared test on categorical features in the dataset, it is found that there is evidence at a significant level of 0.95%, showing that all features have p-values significantly lower than 0.05. The minimum probaility observed of the bunch is 1.56e-10. Rejecting the null hypothesis and supporting that these features are significant. However, it's important to note that not all features comply with the rule of 5 for a Chi-squared test, often failing this criterion in just one category since it is rare. While this could potentially lead to inaccuracies, the overwhelming evidence suggested by the extremely low p-values indicates that these categorical data should be included in the analysis.

![Chi-squared](https://github.com/eliasright/Hotel-Booking-Cancellation-Prediction/assets/151723828/540fa082-d0e6-40e6-ae68-7de11aacae3d)

### E. Preprocessing (Prior Probability)

In every model, continuous features undergo standardization and categorical features are subjected to one-hot encoding, with the omission of the first column. Furthermore, a feature termed 'lead_time_quartiles' is introduced in the transformation pipeline. This feature is derived by segmenting the 'lead_time' into quartiles. Additionally, an analysis of the cancellation odds for each group is integrated into this framework. This involves assessing the historical cancellation patterns within each 'lead_time_quartile'. Specifically, the ratio of bookings that ended in cancellations to those that did not is determined for each quartile. This ratio establishes a prior probability or an estimated likelihood of cancellation correlating with the 'lead_time' period. 

