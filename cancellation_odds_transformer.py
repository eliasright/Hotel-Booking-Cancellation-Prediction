import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CancellationOddsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits=4):
        # Splits are number of range based on quartiles
        self.n_splits = n_splits 
        self.boundaries = None
        self.cancellation_odds = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store the input feature names
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = [f'x{i}' for i in range(X.shape[1])]

        X = pd.DataFrame(X, columns=self.feature_names_in_)

         # Boundaries for splits based n_splits
        self.boundaries = np.quantile(X['lead_time'], [i/self.n_splits for i in range(1, self.n_splits)])

        # Calculate cancellation odds for each split
        for i in range(self.n_splits):
            if i == 0:
                mask = X['lead_time'] <= self.boundaries[0]
            elif i == self.n_splits - 1:
                mask = X['lead_time'] > self.boundaries[-1]
            else:
                mask = (X['lead_time'] > self.boundaries[i-1]) & (X['lead_time'] <= self.boundaries[i])

            group = y[mask]

            odds = group.sum() / len(group) if len(group) > 0 else 0
            self.cancellation_odds[i] = odds

        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)

        # Create a list of conditions based on the boundaries
        conditions = [X['lead_time'] <= self.boundaries[0]] + \
                     [(X['lead_time'] > self.boundaries[i-1]) & (X['lead_time'] <= self.boundaries[i]) for i in range(1, self.n_splits-1)] + \
                     [X['lead_time'] > self.boundaries[-1]]

        # Assign odds of cancellation based on group behaviour
        X['cancellation_prior'] = np.select(conditions, [self.cancellation_odds[i] for i in range(self.n_splits)], default=np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        # Return the names of the output features (For SHAP)
        return np.append(self.feature_names_in_, 'cancellation_prior')










