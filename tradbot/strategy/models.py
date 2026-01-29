"""
Regression Models Module

Linear Regression (OLS) and Logistic Regression for trading strategies.
Based on notebooks/indicators_linreg/8_Regression_all.ipynb
and notebooks/indicators_logreg/5_Overfitting.ipynb
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score


class LinearRegression:
    """
    OLS Linear Regression for price prediction.

    Usage:
        lr = LinearRegression(df, features=['MACD_HIST', 'RSI', 'MFI', 'BB'])
        lr.add_target(shift=5)
        lr.fix_autocorrelation()
        lr.fit()
        lr.validate()
    """

    def __init__(self, df: pd.DataFrame, features: List[str]):
        self.df = df.copy()
        self.features = features
        self.model = None
        self.intercept = None
        self.coefficients = None
        self.p_value = None
        self.r_squared = None

    def add_target(self, shift: int = 5):
        """
        Add target variable (future price change in %).

        Target = (Close[+shift] - Close) / Close * 100
        """
        self.shift = shift
        self.df[f"Close_{shift}"] = self.df["Close"].shift(-shift)
        self.df["Target"] = (
            (self.df[f"Close_{shift}"] - self.df["Close"]) / self.df["Close"] * 100
        )
        return self

    def fix_autocorrelation(self):
        """
        Down-sample to fix autocorrelation in residuals.

        Takes every nth row where n = shift.
        This ensures observations are independent.
        """
        n_before = len(self.df)
        self.df = self.df.iloc[::self.shift].reset_index(drop=True)
        n_after = len(self.df)
        print(f"Down-sampling: {n_before} → {n_after} rows")
        return self

    def fit(self) -> Dict:
        """
        Fit OLS regression model.

        Returns dict with: intercept, coefficients, r_squared, p_value
        """
        # Prepare data
        subset = self.df[self.features + ["Target"]].dropna()
        X = sm.add_constant(subset[self.features])
        y = subset["Target"]

        # Fit model
        self.model = sm.OLS(y, X).fit()

        # Extract results
        self.intercept = self.model.params["const"]
        self.coefficients = self.model.params.drop("const")
        self.r_squared = self.model.rsquared
        self.p_value = self.model.f_pvalue

        # Store predictions
        self.df = self.df.loc[subset.index].copy()
        self.df["Predictions"] = self.model.predict(X)
        self.df["Residuals"] = y - self.df["Predictions"]

        print(self.model.summary())

        return {
            "intercept": self.intercept,
            "coefficients": self.coefficients.to_dict(),
            "r_squared": self.r_squared,
            "p_value": self.p_value,
        }

    def validate(self) -> Dict:
        """
        Validate OLS assumptions and return residual diagnostics.

        Returns dict with residuals for plotting.
        """
        residuals = self.df["Residuals"].dropna()
        predictions = self.df["Predictions"].dropna()

        return {
            "residuals": residuals,
            "predictions": predictions,
            "residuals_mean": residuals.mean(),
            "residuals_std": residuals.std(),
        }

    def get_df(self) -> pd.DataFrame:
        """Return DataFrame with predictions and residuals."""
        return self.df


class LogisticRegression:
    """
    Logistic Regression for binary price direction prediction.

    Usage:
        logreg = LogisticRegression(df, features=['MACD_HIST', 'RSI', 'MFI', 'BB'])
        logreg.add_target(shift=5)
        train_df, test_df = logreg.train_test_split()
        logreg.fit(train_df)
        logreg.predict(test_df)
        logreg.get_auc()
    """

    def __init__(self, df: pd.DataFrame, features: List[str]):
        self.df = df.copy()
        self.features = features
        self.model = None
        self.optimal_shift = None

    def add_target(self, shift: int = 1):
        """
        Add binary target variable.

        Target = 1 if Close[+shift] > Close, else 0
        """
        self.shift = shift
        self.df[f"Close_{shift}"] = self.df["Close"].shift(-shift)
        self.df["Target"] = (self.df[f"Close_{shift}"] > self.df["Close"]).astype(int)
        self.df = self.df.dropna().reset_index(drop=True)
        return self

    def train_test_split(self, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (temporal/chronological).

        Uses time-ordered split to prevent look-ahead bias.
        First 70% of data for training, last 30% for testing.

        Returns (train_df, test_df)
        """
        split_idx = int(len(self.df) * train_size)

        train = self.df.iloc[:split_idx].copy()
        test = self.df.iloc[split_idx:].copy()

        print(f"Train: {len(train)} rows | Test: {len(test)} rows")
        return train, test

    def fit(self, df: pd.DataFrame) -> Dict:
        """
        Fit Logistic Regression on given DataFrame.

        Returns dict with: auc, y_true, y_pred_prob
        """
        # Clean data
        subset = df[self.features + ["Target"]].replace([np.inf, -np.inf], np.nan).dropna()

        X = sm.add_constant(subset[self.features])
        y = subset["Target"]

        # Fit model
        self.model = sm.Logit(y, X).fit(disp=0)

        # Predictions
        y_pred_prob = self.model.predict(X)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Store results
        self.y_true = y
        self.y_pred_prob = y_pred_prob
        self.y_pred = y_pred
        self.auc_score = roc_auc_score(y, y_pred_prob)

        print(f"AUC: {self.auc_score:.4f}")

        return {
            "auc": self.auc_score,
            "y_true": y,
            "y_pred_prob": y_pred_prob,
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict on new data using fitted model.

        Returns predicted probabilities.
        """
        subset = df[self.features].replace([np.inf, -np.inf], np.nan).dropna()
        X = sm.add_constant(subset)
        return self.model.predict(X)

    def explore_shift_auc(self, df: pd.DataFrame, shift_range: range = range(1, 21)) -> pd.DataFrame:
        """
        Find optimal shift by testing AUC across multiple shifts.

        Returns DataFrame with columns: Shift, AUC
        """
        results = []

        for shift in shift_range:
            try:
                # Add target with this shift
                temp_df = df.copy()
                temp_df[f"Close_{shift}"] = temp_df["Close"].shift(-shift)
                temp_df["Target"] = (temp_df[f"Close_{shift}"] > temp_df["Close"]).astype(int)
                temp_df = temp_df.dropna()

                # Fit model
                subset = temp_df[self.features + ["Target"]].replace([np.inf, -np.inf], np.nan).dropna()
                X = sm.add_constant(subset[self.features])
                y = subset["Target"]

                model = sm.Logit(y, X).fit(disp=0)
                y_pred_prob = model.predict(X)
                auc_score = roc_auc_score(y, y_pred_prob)

                results.append({"Shift": shift, "AUC": auc_score})
                print(f"Shift {shift:2d}: AUC = {auc_score:.4f}")

            except Exception as e:
                print(f"Shift {shift}: skipped ({e})")

        results_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
        self.optimal_shift = int(results_df.iloc[0]["Shift"])
        print(f"\nOptimal Shift: {self.optimal_shift}")

        return results_df

    def get_confusion_matrix(self) -> pd.DataFrame:
        """
        Get confusion matrix as DataFrame.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        labels = ["Down (0)", "Up (1)"]
        return pd.DataFrame(cm, index=labels, columns=labels)

    def get_roc_data(self) -> Dict:
        """
        Get ROC curve data for plotting.

        Returns dict with: fpr, tpr, auc
        """
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_prob)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": self.auc_score,
        }

    def get_prediction_distribution(self) -> pd.Series:
        """Get distribution of predicted probabilities."""
        return pd.Series(self.y_pred_prob)
