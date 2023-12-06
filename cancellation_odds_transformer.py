import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CancellationOddsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize attributes
        self.quartiles = None
        self.cancellation_odds = {}

    def fit(self, X, y):
        # Calculate quartiles for lead_time
        self.quartiles = np.quantile(X['lead_time'], [0.25, 0.50, 0.75])

        # Calculate cancellation odds for each quartile
        for i in range(4):
            if i == 0:
                mask = X['lead_time'] <= self.quartiles[0]
            elif i == 3:
                mask = X['lead_time'] > self.quartiles[2]
            else:
                mask = (X['lead_time'] > self.quartiles[i-1]) & (X['lead_time'] <= self.quartiles[i])

            group = y[mask]

            # Calculate the each lead_time quartile cancellation odds
            odds = group.sum() / len(group)
            self.cancellation_odds[i] = odds

        return self

    def transform(self, X):
        # Assign cancellation_prior based on quartile
        conditions = [
            X['lead_time'] <= self.quartiles[0],
            (X['lead_time'] > self.quartiles[0]) & (X['lead_time'] <= self.quartiles[1]),
            (X['lead_time'] > self.quartiles[1]) & (X['lead_time'] <= self.quartiles[2]),
            X['lead_time'] > self.quartiles[2]
        ]

        X['cancellation_prior'] = np.select(conditions, [self.cancellation_odds[i] for i in range(4)], default=np.nan)
        return X

