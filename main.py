import shap
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cancellation_odds_transformer import CancellationOddsTransformer
from bayes_opt import BayesianOptimization
from itertools import combinations
import warnings
from tqdm import tqdm

def main():
    warnings.filterwarnings('ignore')

    # Load the dataset
    file_path = 'Hotel Reservations.csv'
    df = pd.read_csv(file_path)
    X = df.drop('booking_status', axis=1)
    y = df['booking_status'].replace({'Canceled': 1, 'Not_Canceled': 0})

    # Splitting the data into training, testing, and final testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1221)
    X_test, X_final, y_test, y_final = train_test_split(X_test, y_test, test_size=0.5, random_state=1221)

    # Default features chosen for XGBoost
    # Note: 'lead_time' is also included but will specficially hardcoded to be transformed by CancellationOddsTransformer. 
    cols_continuous = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
                        'arrival_month', 'arrival_date', 'no_of_previous_cancellations', 
                        'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests'] 
    cols_categorical = ['type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 
                        'market_segment_type', 'repeated_guest']

    # 1. Baseline model just with the default XGBoost
    run_baseline_model(X_train, X_test, y_train, y_test, X_final, y_final, verbose = 0)

    # 2. Bayesian Optimized XGBoost model
    best_params_xgb, best_n_splits= find_optimized_model(X_train, X_test, y_train, y_test, cols_continuous, cols_categorical, verbose = 0)
    run_optimized_model(X_train, X_test, y_train, y_test, X_final, y_final, best_params_xgb, best_n_splits, cols_continuous, cols_categorical)

    # 3. Bayesian Optimized XGBoost model with feature selections
    best_cols_continuous, best_cols_categorical = test_feature_combinations(X_train, X_test, y_train, y_test, cols_continuous, cols_categorical, verbose = 0)
    best_params_xgb, best_n_splits= find_optimized_model(X_train, X_test, y_train, y_test, cols_continuous, cols_categorical, verbose = 0)
    run_optimized_model(X_train, X_test, y_train, y_test, X_final, y_final, best_params_xgb, best_n_splits, best_cols_continuous, best_cols_categorical)

# Performance print
def evaluation(y_test, y_pred, title):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{title}")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

def run_baseline_model(X_train, X_test, y_train, y_test, X_final, y_final, verbose = 0):
    cols_continuous = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 
                        'arrival_month', 'arrival_date', 'no_of_previous_cancellations', 
                        'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests'] 
    cols_categorical = ['type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 
                        'market_segment_type', 'repeated_guest']

    # Transformer
    continuous_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    cancellation_odds_transformer = CancellationOddsTransformer(n_splits=8)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_transformer, cols_continuous),
            ('cat', categorical_transformer, cols_categorical),
            ('custom', cancellation_odds_transformer, ['lead_time'])
        ])

    # Pipeline fit
    pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', XGBClassifier(random_state=1221))])
    pipeline.fit(X_train, y_train)

    # Evaluation the outcome on testing set and final testing set
    y_pred = pipeline.predict(X_test)
    y_pred_final = pipeline.predict(X_final)
    evaluation(y_test, y_pred, "Base Line Model on X_Test")
    evaluation(y_final, y_pred_final, "Base Line Model on X_Final")

    # SHAP 
    if verbose > 0:
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
        explainer = shap.Explainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_test_transformed)
        feature_names = preprocessor.get_feature_names_out()

        shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
        shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", feature_names=feature_names)

def find_optimized_model(X_train, X_test, y_train, y_test, cols_continuous, cols_categorical, verbose = 0):
    # Optimizing function for XGBoost hyperparameters and n_splits for CancellationOddsTransformer
    def xgb_evaluate(max_depth, learning_rate, n_estimators, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree, n_splits):
        params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'max_delta_step': max_delta_step,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        }

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', continuous_transformer, cols_continuous),
                ('cat', categorical_transformer, cols_categorical),
                ('custom', CancellationOddsTransformer(n_splits=int(n_splits)), ['lead_time'])
            ])

        pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                                      ('smote', SMOTE(random_state=1221)),
                                      ('classifier', XGBClassifier(**params, random_state=1221))])
        pipeline.fit(X_train, y_train)

        # Predict and return F1 score to maximize
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        return f1

    # Transformer
    continuous_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

    # Bayesian Optimizer with F1 as metric
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
    optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=pbounds,
        random_state=1221,
        verbose=0
    )
    optimizer.maximize(init_points=20, n_iter=50)

    # Best parameters
    best_params = optimizer.max['params']
    best_params_xgb = best_params.copy() 
    best_n_splits = int(best_params_xgb.pop('n_splits'))

    # max_depth and n_estimators should be int
    best_params_xgb['max_depth'] = int(best_params_xgb['max_depth'])
    best_params_xgb['n_estimators'] = int(best_params_xgb['n_estimators'])

    if verbose > 0:
        print("\nBest n_splits:", best_n_splits)
        print(f"Best XGBoost params:\n{best_params_xgb}")

    return best_params_xgb, best_n_splits

def run_optimized_model(X_train, X_test, y_train, y_test, X_final, y_final, 
                        best_params_xgb, best_n_splits, cols_continuous, cols_categorical):
    # Transformers
    continuous_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    best_preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_transformer, cols_continuous),
            ('cat', categorical_transformer, cols_categorical),
            ('custom', CancellationOddsTransformer(n_splits=best_n_splits), ['lead_time'])
        ])

    # Pipeline fit
    pipeline = ImbPipeline(steps=[('preprocessor', best_preprocessor),
                                       ('smote', SMOTE(random_state=1221)),
                                       ('classifier', XGBClassifier(**best_params_xgb, random_state=1221))])
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_final = pipeline.predict(X_final)
    evaluation(y_test, y_pred, "Optimized Model on X_test")
    evaluation(y_final, y_pred_final, "Base Line Model on X_Final")

# Function to run xgboost with different sets of features
def run_model_with_features(X_train, X_test, y_train, y_test, continuous_features, categorical_features):
    # Transformers
    continuous_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    cancellation_odds_transformer = CancellationOddsTransformer(n_splits=2)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_transformer, continuous_features),
            ('cat', categorical_transformer, categorical_features),
            ('custom', cancellation_odds_transformer, ['lead_time'])
        ])

    # Pipeline and fit
    pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', XGBClassifier(random_state=1221))])
    pipeline.fit(X_train, y_train)

    # Fit and return F1 as metric
    y_pred = pipeline.predict(X_test)
    return f1_score(y_test, y_pred)

# Generate different combinations of features (minimum 10)
def test_feature_combinations(X_train, X_test, y_train, y_test, cols_continuous,cols_categorical, verbose=0):
    best_f1_score = 0
    best_features = []

    all_features = cols_continuous + cols_categorical
    min_combination_size = 10 # Arbitrarly chosen as 10 to reduce computational power of this method
    total_combinations = sum(1 for r in range(min_combination_size, len(all_features) + 1) for _ in combinations(all_features, r))

    # Cycle through each combinations to find the one with the highest F1 score on test set
    with tqdm(total=total_combinations, desc="Testing Combinations") as pbar:
        for i in range(min_combination_size, len(all_features) + 1):
            for combo in combinations(all_features, i):
                selected_features = list(combo)
                selected_continuous = [f for f in selected_features if f in cols_continuous]
                selected_categorical = [f for f in selected_features if f in cols_categorical]

                X_train_subset = X_train[selected_features + ['lead_time']]
                X_test_subset = X_test[selected_features + ['lead_time']]

                f1 = run_model_with_features(
                    X_train_subset, X_test_subset, 
                    y_train, y_test, 
                    selected_continuous, selected_categorical
                )

                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_features = selected_features + ['lead_time']

                pbar.update(1)

    best_cols_continuous = [f for f in best_features if f in cols_continuous]
    best_cols_categorical = [f for f in best_features if f in cols_categorical]

    if verbose > 0:
        print(f"Best feature set: {best_features}")
        print(f"Associated F1 score: {best_f1_score}")

    return best_cols_continuous, best_cols_categorical

if __name__ == '__main__':
    main()
