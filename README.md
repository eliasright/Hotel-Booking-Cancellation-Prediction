# Hotel Booking Cancellation Prediction
## Files
Absolutely, here's a confident, first-person description of the files for my GitHub repository:

1. **Images/**: This directory contains all the image outputs from main.py and data_exploration.py.

2. **.gitattributes**: A configuration file I use to manage specific attributes of the paths in my repository.

3. **Hotel Reservations.csv**: This is the dataset file containing the hotel reservation data I analyze.

4. **data_exploration.py**: My Python script for performing the initial exploration and analysis of the dataset.

5. **README.md**: A Markdown file where I provide a comprehensive overview, documentation, and instructions for the project.

7. **cancellation_odds_transformer.py**: A Python script dedicated to processing and analyzing the odds of hotel cancellations.

8. **main.py**: The main Python script of the project, executing the core workflow and data analysis.

9. **requirements.txt**: This file lists all the Python dependencies necessary for the project, ensuring a consistent setup environment.
## How to run

1. **Download the Repository**: Clone or download the repository from GitHub to your local machine.

2. **Install Requirements**: Run `pip install -r requirements.txt` to install the required Python packages listed in `requirements.txt`.

3. **Run the Main Script**: Execute the main script of the project by running `python main.py`.

## Motivation
The motivation behind this test XGBoost in predict whether a guest would cancel a booking. Then try to finetune the model to improve upon the metrics: Accuracy, Precision, Recall, and F1. The predictive model will be trained using a training set, finetune on a validation set (called X_test in the code), and testing set (called X_final in the code). This way, I can improve on the model and use the validation set to guide me in improving the model. Just to make sure I am not overfitting to the validation set, the final testing set will be only tested at the end to see how well the model performed.

The dataset is split into 3 type of features:
- Continuous features
- Categorical features
- `lead_time` -> `prior_probability`

Here `lead_time` refers to the duration between the booking date and the date of arrival. A custom transformer will be utilized to categorize customers into different clusters based on their `lead_time`. Subsequently, for each of these clusters, the likelihood of cancellation is calculated. For instance, if a cluster comprises 6 people and 4 out of these 6 individuals cancel while 2 do not, a prior probability of 4/6 is assigned to that cluster. This probability will then be applied to any new data that falls within the same `lead_time` range for that particular cluster.

Initially, in Model 1, the `lead_time` is divided into four quartiles, representing the lower 25%, 25-50%, 50-75%, and 75-100% ranges. However, to enhance the model's effectiveness, Bayesian optimization will be employed later to determine an optimal split. This approach may result in a different partitioning of the 'lead_time', such as into two or eight segments, instead of the initial four

There will be a total of 3 models:
- Baseline model: XGBoost with default hyperparameters
- Bayesian Optimized XGBoost
- Bayesian Optimized XGBoost with feature selections

**Dataset:** Hotel Reservations Dataset

**Link:** https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset/data

**NOTE:** Dataset is considered to be data collected over a year and a half.

## Project Overview

1. [Data Overview and Preprocessing](#1-data-overview-and-preprocessing)
2. [Models Performance](#2-models-performance)
4. [Conclusion](#3-conclusion)

### Summary of Findings

- Model 2 improved upon Model 1 with the use of Bayesian Optimization.
- Model 3 improved upon Model 2 on the testing set but worsen for the validation set due to bad assumption in feature selection.
- Only a small percentage increment in performance but in real-world settings with thousands of bookings, even small percentage improvements in predicting customer cancellations can have a substantial impact.
- Future finetuning could be explored

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

Note: `lead_time_quartiles` is what was mentioned in [Motivation](#motivation)

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

In every model, continuous features undergo standardization and categorical features are subjected to one-hot encoding, with the omission of the first column. Furthermore, a feature termed `lead_time_quartiles` is introduced in the transformation pipeline. This feature is derived by segmenting the `lead_time` into quartiles. Additionally, an analysis of the cancellation odds for each group is integrated into this framework. This involves assessing the historical cancellation patterns within each `lead_time_quartile`. Specifically, the ratio of bookings that ended in cancellations to those that did not is determined for each quartile. This ratio establishes a prior probability or an estimated likelihood of cancellation correlating with the `lead_time` period. 

## 2. Models performance

### Model 1: Baseline Model

Model 1 serves as the baseline model, establishing foundational benchmarks in terms of key metrics: Accuracy, Precision, Recall, and F1 Score. In this context, the F1 Score is particularly valuable due to its relevance in assessing customer cancellation scenarios, among other factors. Future models will be optimized using F1 because it balances precision and recall, providing a more comprehensive measure of a model's performance, especially for customer cancellations where both false positives and false negatives are critical.

Data were transformed
- Standardization
- One Hot Encoded with 1 dummpy dropped
- 'lead_time' CancellationOddsTransformation

XGBoost
- Default parameters

#### Results
The results on the validation set `X_test`
```
Accuracy: 0.8851314096673406
Precision: 0.8512639623750735
Recall: 0.7956043956043956
F1 Score: 0.8224936097699517
```

The results on the testing set `X_final `
```
Accuracy: 0.8948915839764793
Precision: 0.8612143742255266
Recall: 0.7997698504027618
F1 Score: 0.8293556085918854
```

#### SHAP analysis

I conducted a SHAP (SHapley Additive explanations) analysis to evaluate the impact of each feature on the predictive model. The analysis reveals how different features influence the model's output, with the SHAP value indicating the magnitude and direction of the feature's effect. The visualization shows that features such as `custom_lead_time` (which is the `lead_time_quartiles` cancellation prior probabilit) and `num_no_of_special_requests` have a high positive impact on the model's predictions, whereas `cat_market_segment_type_Complementary` and `num_no_of_children` exhibit a lower impact.

The SHAP output illustrates the relationship between feature values and their impact on the target variable booking_status, where 1 represents a cancellation and 0 represents a non-cancellation. The color scale indicates the feature value, with red denoting high values and blue indicating low values within their respective ranges. For instance, a red dot positioned to the right of the x-axis under custom_lead_time suggests that higher lead times are strongly associated with an increased likelihood of cancellation. 

Notably, 'custom_lead_time' emerges as the most influential feature in the SHAP analysis; it consistently shows that higher values, indicated by red, are positively correlated with an increased likelihood of cancellation, confirming its predictive importance in the model. Conversely, a higher number of special requests, which appear as red on the plot and are positioned towards the negative side of the x-axis, are negatively correlated with cancellation. This suggests that as the number of special requests increases, the probability of the guest not cancelling their booking also increases. 

### Model 2: Bayesian Optimization
In this model, we employed Bayesian Optimization due to the high number of hyperparameters in XGBoost. This approach is computationally faster compared to methods like grid search

Data were transformed
- Standardization
- One Hot Encoded with 1 dummpy dropped
- 'lead_time' CancellationOddsTransformation

XGBoost
- Bayesian Optimization with bounds
```
pbounds = {
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (50, 1000),
        'gamma': (0, 5),
        'min_child_weight': (0, 5),
        'max_delta_step': (0, 5),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'n_splits': (2, 8)  
    }
```

Note: n_splits is the hyperparameters of of the custom transformer `CancellationOddsTransformer`

#### Results
The results on the validation set `X_test`
```
Accuracy: 0.8891747840470502
Precision: 0.845150311968236
Recall: 0.8186813186813187
F1 Score: 0.8317052749092938
```

The results on the testing set `X_final `
```
Accuracy: 0.8981991914737228
Precision: 0.8583535108958837
Recall: 0.8158803222094362
F1 Score: 0.8365781710914455
```

#### Changes
From the baseline model

The Validation set
- Accuracy improved by approximately 0.331%
- Precision decreased slightly by approximately -0.286%
- Recall improved by approximately 1.611%
- F1 Score improved by approximately 0.722%

The testing set (Where performance will be measured)
- Accuracy improved by approximately 0.404%
- Precision decreased slightly by approximately -0.611%
- Recall improved by approximately 2.308%
- F1 Score improved by approximately 0.921%

We see an overall improvement in the model's performance, especially in the critical metrics of recall and F1 Score, which are essential in contexts where the balance between false positives and false negatives is important. The improvements in recall on both the validation and testing sets are particularly notable, suggesting that the model is becoming more adept at identifying true positives.

### Model 3: Bayesian Optimization w\ feature selections
I selected a subset of features, choosing at least 10 out of a possible 15, and assessed their performance using the F1 score metric on the validation set. The top-performing features were subsequently incorporated into the Bayesian Optimization process to refine the model.

Feature selection
- Subset of features were chosen

Data were transformed
- Standardization
- One Hot Encoded with 1 dummpy dropped
- 'lead_time' CancellationOddsTransformation

XGBoost
- Bayesian Optimization with bounds
```
pbounds = {
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'n_estimators': (50, 1000),
        'gamma': (0, 5),
        'min_child_weight': (0, 5),
        'max_delta_step': (0, 5),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.5, 1),
        'n_splits': (2, 8)  
    }
```

#### Results
The results on the validation set `X_test`
```
Accuracy: 0.8866017276235986
Precision: 0.8435179897201599
Recall: 0.8115384615384615
F1 Score: 0.8272192663119574
```

The results on the testing set `X_final `
```
Accuracy: 0.898566703417861
Precision: 0.8589588377723971
Recall: 0.8164556962025317
F1 Score: 0.8371681415929204
```

#### Changes
**From the baseline model to model 3**
The Validation set
- Accuracy improved slightly by approximately 0.166%
- Precision decreased slightly by approximately 0.91%
- Recall improved by approximately 2.00%
- F1 Score improved by approximately 0.575%

The testing set (Where performance will be measured)
- Accuracy improved by approximately 0.411%
- Precision decreased slightly by approximately 0.262%
- Recall improved by approximately 2.09%
- F1 Score improved by approximately 0.942%

We see a noteworthy progression in model performance from the baseline to Model 3. The improvements in accuracy and recall for both the validation and testing sets are particularly significant. An increase in recall by approximately 2.00% in the validation set and 2.09% in the testing set indicates the model's enhanced capability to identify true positives. Despite a slight decrease in precision, the substantial improvements in recall and F1 Score suggest that Model 3 is more effective in balancing false positives and negatives

**From the model 2 to model 3**
For the validation set (X_test):
- Accuracy decreased by approximately -0.257%
- Precision decreased by approximately -0.163%
- Recall decreased by approximately -0.714%
- F1 Score decreased by approximately -0.449%

The testing set (Where performance will be measured)
- Accuracy improved slightly by approximately 0.037%
- Precision improved slightly by approximately 0.061%
- Recall improved slightly by approximately 0.058%
- F1 Score improved slightly by approximately 0.059%

In transitioning from Model 2 to Model 3, I implemented feature selection based on XGBoost's default hyperparameters, followed by Bayesian optimization on the selected features. This approach, however, was grounded in an assumption that may not have been optimal. The deterioration in performance metrics on the validation set, as evidenced by the decreases in accuracy, precision, recall, and F1 Score, suggests that my assumption during feature selection and subsequent optimization might have inadvertently led to a model that was less effective at generalizing to validation data. This could be indicative of overfitting, where the model, though finely tuned on certain features, lost some of its predictive robustness on the validation set.

## 3. Conclusion
From the baseline Model 1 to Model 3 demonstrated an overall improvements, particularly in recall and F1 Score, crucial for predicting customer cancellations. Despite a slight decline in precision, these models showed a better balance in handling false positives and negatives, essential for practical applications.

Transitioning from Model 2 to Model 3 showed an interesting trend: While the validation set's performance slightly declined, indicating challenges in the model's generalizability, the testing set exhibited improvements. While Model 3 achieved success in terms of the project's objective to blind test the testing set "X_final", the mixed results observed in the validation set suggest this improvement may be attributable to chance, necessitating further investigation to confirm its validity.

While the improvements were generally modest, often less than one percent and at most two percent, these incremental gains are significant in practical scenarios. In real-world settings with thousands of bookings, even small percentage improvements in predicting customer cancellations can have a substantial impact.

Future improvements
- Move beyond XGBoost's default settings in feature selections for better generalization
- Implement methods to reduce overfitting
- Experiment with a wider range of hyperparameters
- Incorporate more diverse data for enhanced real-world applicability. Through feature interactions or more data collection
- Predict with other models
