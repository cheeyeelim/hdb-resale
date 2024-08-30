"""Contains functions for hdb_resale specific model operations."""
import logging
import sys
import time

import cloudpickle
import lightgbm as lgbm
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import max_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from yellowbrick.features import rank2d
from yellowbrick.model_selection import feature_importances
from yellowbrick.regressor import prediction_error, residuals_plot
from yellowbrick.target.feature_correlation import feature_correlation

logger = logging.getLogger(__name__)


class MultiTreeEnsembleRegressor(VotingRegressor):
    """Class for an ensemble of multiple tree-based models.

    Currently this regressor uses VotingRegressor to combine outputs from
    3 tree-based models: xgboost, lightgbm and sklearn's random forest.

    Each tree-based model uses default parameters without any customisation or tuning,
    except random forest is set to max_depth = 10 to prevent overfitting.

    Parameters
    ----------
    weights: List[float]
        Weight of each model contributing to final prediction.
        Default to equal weight for all models.
    n_jobs: int
        The number of jobs to run in parallel.
        Parallel n_jobs should be set at individual estimator level.
        Setting at VotingRegressor will lead to slower fit time.
        Default to -1, i.e. using all cores.
    random_state: int
        Set the random seed of underlying models.
        Default to 6.

    Attributes
    ----------
    random_state:
        Random seed used for model fitting.
    feature_importances_: ndarray, shape (n_features,)
        Feature importance.
    """

    def __init__(self, weights=[1, 1, 1], n_jobs=-1, random_state=6) -> None:
        """Initialise model."""
        self.random_state = random_state

        # Setup underlying model.
        # NOTE - set max_depth to prevent random forest overfitting
        # NOTE - LGBM & XGB are well-regularized by default, compared to random forest
        model_one = RandomForestRegressor(max_depth=10, random_state=random_state, n_jobs=n_jobs)
        model_two = lgbm.LGBMRegressor(random_state=random_state, n_jobs=n_jobs)
        model_three = xgb.XGBRegressor(random_state=random_state, n_jobs=n_jobs)

        estimators = [
            ("random_forest", model_one),
            ("lightgbm", model_two),
            ("xgboost", model_three),
        ]

        super().__init__(estimators=estimators, weights=weights, n_jobs=n_jobs)

    def fit(self, X, y):
        """Fit the ensemble model.

        Parameters
        ----------
        X: pd.DataFrame, array-like, shape (n_samples, n_features)
            Data with feature values.
        y: pd.Series, array-like, shape (n_samples,)
            Data with target values.

        Returns
        -------
        self: object
            Returns self.
        """
        return super().fit(X, y)

    def predict(self, X: pd.DataFrame):
        """Predict labels using the fitted ensemble model.

        Parameters
        ----------
        X: pd.DataFrame, array-like, shape (n_samples, n_features)
            Data with feature values.

        Returns
        -------
        y: np.ndarray, shape (n_samples,)
            The predicted labels.
        """
        return super().predict(X)

    @property
    def feature_importances_(self):
        """Get overall feature importance by averaging feature importance from each estimator.

        Returns
        -------
        feat_imp_series: np.ndarray, shape (n_features,)
            Average feature importance from all estimators.
        """
        # Get feature importance from each estimator
        feat_importance = dict()
        for i, est in enumerate(self.estimators_):
            feat_importance[self.estimators[i][0]] = est.feature_importances_

        feat_imp_df = pd.DataFrame(feat_importance)

        # Normalize values to sum to 1
        feat_imp_df = feat_imp_df / feat_imp_df.sum()

        # Multiply by the weights of each estimator
        feat_imp_df = feat_imp_df.mul(self.weights)

        # Sum feature importance per feature $ renormalize
        feat_imp_series = feat_imp_df.sum(axis=1)
        feat_imp_series = feat_imp_series / feat_imp_series.sum()

        return feat_imp_series.to_numpy()


def train_model(X, y, model):
    """Train a model as specified.

    Parameters
    ----------
    X: pd.DataFrame, array-like, shape (n_samples, n_features)
        Data with feature values.
    y: pd.Series, array-like, shape (n_samples,)
        Data with target values.
    model : VotingRegressor
        Defined sklearn VotingRegressor ensemble models.

    Returns
    -------
    model : VotingRegressor
        Fitted/trained sklearn VotingRegressor ensemble models.
    """
    model = model.fit(X, y)

    return model


def diagnose_model(X, y, model, random_state):
    """Diagnose the training & performance of the model.

    NOTE this takes a very long time to run (~ 20 mins depending on data & model settings).

    Parameters
    ----------
    X: pd.DataFrame, array-like, shape (n_samples, n_features)
        Data with feature values.
    y: pd.Series, array-like, shape (n_samples,)
        Data with target values.
    model : VotingRegressor
        Fitted/trained sklearn VotingRegressor ensemble models.
    random_state: int
        Set the random seed of underlying non-deterministic algorithms.

    Returns
    -------
    cv_res: pd.DataFrame
        Evaluate metrics calculated for each cross validation.
    size: pd.DataFrame
        Size (in terms of memory) of each ensemble models.
    diag_fig: matplotlib.figure.Figure
        Diagnostics plots. Generated using yellowbricks.
    """
    # Get error metrics from cross validations
    cv_res = get_cv_error(X=X, y=y, model=model, random_state=random_state)

    # Get actual fitted model sizes
    # For performance considerations
    size = get_model_size(model=model)

    # Get various diagnostic plots - both pre and post predictions
    # NOTE Slowest section in terms of computation
    diag_fig = get_diag_plot(X=X, y=y, model=model, random_state=random_state)

    return cv_res, size, diag_fig


def get_cv_error(X, y, model, random_state):
    """Get error metrics to assess model performance using cross validation.

    Parameters
    ----------
    X: pd.DataFrame, array-like, shape (n_samples, n_features)
        Data with feature values.
    y: pd.Series, array-like, shape (n_samples,)
        Data with target values.
    model : VotingRegressor
        Fitted/trained sklearn VotingRegressor ensemble models.
    random_state: int
        Set the random seed of underlying non-deterministic algorithms.

    Returns
    -------
    cv_res : pd.DataFrame
        Error metrics from cross validations.
    """
    # Perform cross validation to check error metrics
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    mse_res_list = []
    mape_res_list = []
    max_res_list = []
    r2_res_list = []
    elapsed_time_list = []
    for train_index, test_index in cv.split(X):
        X_sub_train = X.loc[train_index, :]
        y_sub_train = y.loc[train_index]
        X_sub_test = X.loc[test_index, :]
        y_sub_test = y.loc[test_index]

        start_time = time.time()

        # Train model & generate model predictions
        model = model.fit(X_sub_train, y_sub_train)
        pred_res = model.predict(X_sub_test)

        elapsed_time = time.time() - start_time

        # Assess performance
        true_res = y_sub_test.reset_index(drop=True)

        mse_res = mean_squared_error(true_res, pred_res)
        mape_res = mean_absolute_percentage_error(true_res, pred_res)
        max_res = max_error(true_res, pred_res)
        r2_res = r2_score(true_res, pred_res)

        mse_res_list.append(mse_res)
        mape_res_list.append(mape_res)
        max_res_list.append(max_res)
        r2_res_list.append(r2_res)
        elapsed_time_list.append(elapsed_time)

    cv_res_dict = {
        "mse": mse_res_list,
        "mape": mape_res_list,
        "max": max_res_list,
        "r2": r2_res_list,
        "elapsed_time": elapsed_time_list,
    }

    cv_res = pd.DataFrame(cv_res_dict)

    return cv_res


def get_model_size(model):
    """Get sizes (in bytes) of overall and individual model components.

    Works for sklearn VotingRegressor ensemble models.

    Parameters
    ----------
    model : VotingRegressor
        Fitted/trained sklearn VotingRegressor ensemble models.

    Returns
    -------
    size : pd.DataFrame
        Absolute and relative model sizes.
    """
    overall_model_size = sys.getsizeof(cloudpickle.dumps(model))

    component_model_name = []
    component_model_size = []
    for i in range(len(model.estimators_)):
        component_model_name.append(model.estimators_[i].__class__.__name__)
        component_model_size.append(sys.getsizeof(cloudpickle.dumps(model.estimators_[i])))

    size = pd.DataFrame(
        {
            "name": [model.__class__.__name__] + component_model_name,
            "act_size": [overall_model_size] + component_model_size,
        }
    )
    size["pct_size"] = size["act_size"] / overall_model_size

    return size


def get_diag_plot(X, y, model, random_state):
    """Get diagnostic plots on various pre- and post-prediction metrics.

    Diagnostic tests include :
    - Pairwise feature Spearman rank correlation
    - Target-feature mutual information correlation
    - PCA dimensional reduction of features, with biplots
    - tSNE dimensional reduction of features
    - prediction errors, i.e. actuals vs predicted values
    - residual distribution, i.e. predicted values vs residuals
    - feature importance ranked barchart

    NOTE this function is slow due to multiple internal cross validation steps.

    Powered by `yellowbrick`.

    Parameters
    ----------
    X: pd.DataFrame, array-like, shape (n_samples, n_features)
        Data with feature values.
    y: pd.Series, array-like, shape (n_samples,)
        Data with target values.
    model : VotingRegressor
        Fitted/trained sklearn VotingRegressor ensemble models.
    random_state: int
        Set the random seed of underlying non-deterministic algorithms.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Plot of subplots of various diagnostic tests.
    """
    # Split data into train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Perform diagnostics using yellowbrick.
    fig, axs = plt.subplots(4, 2, figsize=(16, 22), facecolor="white")

    rank2d(
        X,
        algorithm="spearman",
        show=False,
        ax=axs[0, 0],
    )

    feature_correlation(
        X,
        y,
        method="mutual_info-regression",
        show=False,
        ax=axs[0, 1],
    )

    # # Breaking bug due to deprecated matplotlib API
    # pca_decomposition(
    #     X,
    #     y,
    #     scale=True,
    #     proj_features=True,
    #     show=False,
    #     ax=axs[1, 0],
    # )

    # # Breaking bug due to deprecated matplotlib API
    # manifold_embedding(
    #     X,
    #     y,
    #     manifold="tsne",
    #     n_neighbors=5,
    #     show=False,
    #     ax=axs[1, 1],
    # )

    prediction_error(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        show=False,
        ax=axs[2, 0],
    )

    residuals_plot(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        show=False,
        ax=axs[2, 1],
    )

    feature_importances(
        model,
        X_train,
        y_train,
        show=False,
        ax=axs[3, 0],
    )

    return fig
