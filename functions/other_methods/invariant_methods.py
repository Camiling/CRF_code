import numpy as np
import pandas as pd
from anchorboosting import AnchorBooster

from typing import Dict, Union

import math
import numpy as np
import pandas as pd
from sklearn import metrics
from numpy.random import default_rng
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.base import BaseEstimator, ClassifierMixin

# IRF taken from https://colab.research.google.com/drive/1sEhz7BlSq1zXPvqFzsOuT78IP94i3NBH#scrollTo=KkafPZ0jSYlN
# Anchor boosting downloaded as python package

# ================================
# Helper functions
# ================================

def std_agg(cnt, s1, s2):
    try:
        return math.sqrt((s2 / cnt) - (s1 / cnt) ** 2)
    except Exception:
        # when numerical issues lead to negative inside sqrt
        return 0.0


def check_min_sample_periods_dict(count_dict, min_sample_periods):
    """
    Check if all periods listed in the dictionary contain at
    least min_sample_periods examples.
    """
    for key in count_dict.keys():
        if count_dict[key] < min_sample_periods:
            return False
    return True


def check_min_sample_periods(X, time_column, min_sample_periods):
    """
    Check if all periods contained in a dataframe for a certain time_column
    contain at least min_sample_periods examples.
    """
    return (X[time_column].value_counts() >= min_sample_periods).prod()


def initialize_period_dict(periods):
    """
    Initialize the period dict with all the distinct periods as key, and
    the needed keys to calculate the loss function.
    """
    period_dict: Dict[Union[str, int], Dict[str, float]] = {}
    for period in periods:
        period_dict[period] = {
            "count": 0.0,
            "sum": 0.0,
            "squared_sum": 0.0,
        }
    return period_dict


def fill_right_dict(periods, target, weights, right_dict):
    """
    The right dict is the one that starts with all the data before we perform
    any split. This functions fills it with the initial condition.
    """
    for period in np.unique(periods):
        period_filter = (periods == period)
        period_weights = weights[period_filter]
        right_dict[period]["count"] = period_weights.sum()
        right_dict[period]["sum"] = (target[period_filter] * period_weights).sum()
        right_dict[period]["squared_sum"] = (
            (target[period_filter] ** 2) * period_weights
        ).sum()
    return right_dict


def std_score_by_period(right_dict, left_dict, norm=False):
    """
    Calculate the standard deviation score by period given two dictionaries that
    charactize the left and right leaf after the potential split.
    """
    current_score = []
    for key in right_dict.keys():
        left_std = std_agg(
            left_dict[key]["count"],
            left_dict[key]["sum"],
            left_dict[key]["squared_sum"],
        )
        right_std = std_agg(
            right_dict[key]["count"],
            right_dict[key]["sum"],
            right_dict[key]["squared_sum"],
        )

        if norm:
            total_count = left_dict[key]["count"] + right_dict[key]["count"]
            current_score.append(
                left_std * (left_dict[key]["count"] / total_count)
                + right_std * (right_dict[key]["count"] / total_count)
            )
        else:
            current_score.append(
                left_std * left_dict[key]["count"]
                + right_std * right_dict[key]["count"]
            )
    return current_score


def gini_impurity_score_by_period(right_dict, left_dict, verbose=False):
    """
    Calculate the gini impurity score by period given two dictionaries that
    charactize the left and right leaf after the potential split.
    """
    current_score = []
    for key in right_dict.keys():
        left_proba = left_dict[key]["sum"] / float(left_dict[key]["count"])
        left_gini = 1 - ((1 - left_proba) ** 2 + (left_proba) ** 2)

        right_proba = right_dict[key]["sum"] / float(right_dict[key]["count"])
        right_gini = 1 - ((1 - right_proba) ** 2 + (right_proba) ** 2)

        total_count = left_dict[key]["count"] + right_dict[key]["count"]
        current_score.append(
            left_gini * (left_dict[key]["count"] / total_count)
            + right_gini * (right_dict[key]["count"] / total_count)
        )
    return current_score


def aggregate_dict(right_dict, left_dict):
    agg_right_dict = {"count": 0.0, "sum": 0.0}
    agg_left_dict = {"count": 0.0, "sum": 0.0}

    for key in right_dict.keys():
        agg_right_dict["count"] += right_dict[key]["count"]
        agg_right_dict["sum"] += right_dict[key]["sum"]

        agg_left_dict["count"] += left_dict[key]["count"]
        agg_left_dict["sum"] += left_dict[key]["sum"]

    return agg_right_dict, agg_left_dict


def changing_rate(env_right_dict, env_left_dict, pos_class):
    total_positive = env_left_dict["sum"] + env_right_dict["sum"]
    total_negative = (
        env_left_dict["count"]
        + env_right_dict["count"]
        - total_positive
    )

    if pos_class:
        return (env_left_dict["sum"] + 0.5) / (total_positive + 1.0)
    else:
        return (
            env_left_dict["count"] - env_left_dict["sum"] + 0.5
        ) / (total_negative + 1.0)


def invariant(env_right_dict, env_left_dict):
    pos_changing_rate = changing_rate(env_right_dict, env_left_dict, pos_class=True)
    neg_changing_rate = changing_rate(env_right_dict, env_left_dict, pos_class=False)
    return pos_changing_rate / neg_changing_rate


def invariance_loss_function(right_dict, left_dict):
    invariants = []
    for env in right_dict.keys():
        invariants.append(invariant(right_dict[env], left_dict[env]))
    return (max(invariants) / min(invariants)) - 1.0


def gini_invariance_penalty_score(
    right_dict,
    left_dict,
    invariance_penalty=10.0,
    verbose=False,
):
    agg_right_dict, agg_left_dict = aggregate_dict(right_dict, left_dict)

    # use same machinery as gini_impurity_score_by_period
    overall_gini = gini_impurity_score_by_period(
        {"dummy": agg_right_dict},
        {"dummy": agg_left_dict}
    )[0]

    invariance_loss = invariance_loss_function(right_dict, left_dict)
    if verbose:
        print(
            "Overall_gini: {}, invariance_penalty: {}, invariance_loss: {}".format(
                overall_gini, invariance_penalty, invariance_loss
            )
        )

    return overall_gini + (invariance_penalty * invariance_loss)


def score_by_period(
    right_dict,
    left_dict,
    criterion="std",
    period_criterion="avg",
    invariance_penalty=10.0,
    verbose=False,
):
    """
    Switches between the different criterion for loss function and its
    aggregation.
    """
    if criterion == "gini":
        current_score = gini_impurity_score_by_period(
            right_dict, left_dict, verbose=verbose
        )
    elif criterion == "std":
        current_score = std_score_by_period(right_dict, left_dict)
    elif criterion == "std_norm":
        current_score = std_score_by_period(right_dict, left_dict, norm=True)
    elif criterion == "gini_invariance_penalty":
        current_score = gini_invariance_penalty_score(
            right_dict,
            left_dict,
            invariance_penalty=invariance_penalty,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    if verbose:
        print(f"Score {criterion} by period: {current_score}")

    if period_criterion == "avg":
        return np.mean(current_score)
    else:
        return np.max(current_score)


def impurity_decrease_by_period(
    right_dict,
    left_dict,
    total_sample,
    period_criterion="avg",
    verbose=False,
):
    impurity_decreases = []

    for key in right_dict.keys():
        total_count = left_dict[key]["count"] + right_dict[key]["count"]
        total_positive = left_dict[key]["sum"] + right_dict[key]["sum"]
        p_positive = total_positive / total_count

        previous_impurity = 1.0 - ((1 - p_positive) ** 2 + (p_positive) ** 2)

        left_proba = left_dict[key]["sum"] / float(left_dict[key]["count"])
        left_gini = 1.0 - ((1 - left_proba) ** 2 + (left_proba) ** 2)

        right_proba = right_dict[key]["sum"] / float(right_dict[key]["count"])
        right_gini = 1.0 - ((1 - right_proba) ** 2 + (right_proba) ** 2)

        score = (
            left_gini * (left_dict[key]["count"] / total_count)
            + right_gini * (right_dict[key]["count"] / total_count)
        )

        impurity_decrease = (
            total_count / total_sample[key] * (previous_impurity - score)
        )

        if verbose:
            print(
                "Period: {}, score:{}, left: {}, right: {}, impurity decrease: {}".format(
                    key, score, left_gini, right_gini, impurity_decrease
                )
            )

        impurity_decreases.append(impurity_decrease)

    if period_criterion == "avg":
        return np.mean(impurity_decreases)
    else:
        return np.min(impurity_decreases)


def split_time_index_column_into_n_segments(df, n_segments, time_index_column):
    """
    Splits the time index / stamp column into n segments.
    """
    return pd.cut(
        df[time_index_column],
        bins=n_segments,
        labels=[str(i) for i in range(n_segments)],
    ).astype(str)


def generate_n_segments_columns(df, n_segments, time_index_column):
    """
    Generates n segment columns based in a time_index_column able to order the
    examples from older to newer.
    """
    random_segments_columns = []
    for segment in range(0, n_segments):
        segment_column_name = f"time_column_segment_{str(segment)}"
        df[segment_column_name] = split_time_index_column_into_n_segments(
            df, segment + 1, time_index_column
        )
        random_segments_columns.append(segment_column_name)

    return random_segments_columns


# ================================
# TimeForestClassifier
# ================================

class TimeForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Time Forest Classifier Estimator.
    """

    def __init__(
        self,
        n_estimators=5,
        time_column="period",
        max_depth=5,
        min_sample_periods=100,
        max_features="auto",
        bootstrapping=True,
        criterion="gini",
        period_criterion="avg",
        invariance_penalty=10.0,
        min_impurity_decrease=0.0,
        n_jobs=-1,
        multi=True,
        random_segments=None,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.time_column = time_column
        self.min_sample_periods = min_sample_periods
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.multi = multi
        self.bootstrapping = bootstrapping
        self.criterion = criterion
        self.period_criterion = period_criterion
        self.invariance_penalty = invariance_penalty
        self.min_impurity_decrease = min_impurity_decrease
        self.random_segments = random_segments
        self.random_state = random_state
        self.rng = default_rng(self.random_state)

    def fit(self, X, y, sample_weight=None, verbose=False):
        """
        Learns the classifier model from the training data.
        """
        if self.n_jobs <= 0:
            self.n_jobs = cpu_count() - 2

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            X.columns = [*X.columns[:-1], self.time_column]
        if isinstance(y, pd.Series):
            y = y.values

        if isinstance(self.random_segments, int):
            X["target_"] = y
            X.sort_values(by=self.time_column, inplace=True)
            X["time_index"] = range(1, len(X) + 1)
            self.random_segments_columns = generate_n_segments_columns(
                X, self.random_segments, "time_index"
            )

            y = X["target_"].values
            X.drop(columns=["time_index", self.time_column, "target_"], inplace=True)
        elif self.random_segments is None:
            self.random_segments = 1
            self.random_segments_columns = [self.time_column]
        else:
            self.random_segments_columns = self.random_segments
            self.random_segments = len(self.random_segments_columns)

        self.train_target_proportion = np.mean(y)
        self.classes_ = np.unique(y)

        self.n_estimators_ = []
        self.selected_time_columns = [
            self.random_segments_columns[
                self.rng.integers(0, self.random_segments)
            ]
            for _ in range(self.n_estimators)
        ]

        features = [
            col for col in X.columns if col not in self.random_segments_columns
        ]

        if not self.multi:
            self.n_estimators_ = [
                _RandomTimeSplitTree(
                    X[features + [self.selected_time_columns[i]]],
                    y,
                    min_sample_periods=self.min_sample_periods,
                    max_depth=self.max_depth,
                    bootstrapping=self.bootstrapping,
                    sample_weight=sample_weight,
                    time_column=self.selected_time_columns[i],
                    row_indexes=[],
                    verbose=verbose,
                    max_features=self.max_features,
                    criterion=self.criterion,
                    period_criterion=self.period_criterion,
                    invariance_penalty=self.invariance_penalty,
                    min_impurity_decrease=self.min_impurity_decrease,
                    total_sample=X[self.selected_time_columns[i]].value_counts().to_dict(),
                    random_state=i + self.random_state,
                )
                for i in range(self.n_estimators)
            ]
        else:
            self.n_estimators_ = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_RandomTimeSplitTree)(
                    X[features + [self.selected_time_columns[i]]],
                    y,
                    min_sample_periods=self.min_sample_periods,
                    max_depth=self.max_depth,
                    bootstrapping=self.bootstrapping,
                    sample_weight=sample_weight,
                    time_column=self.selected_time_columns[i],
                    row_indexes=[],
                    verbose=verbose,
                    max_features=self.max_features,
                    period_criterion=self.period_criterion,
                    invariance_penalty=self.invariance_penalty,
                    min_impurity_decrease=self.min_impurity_decrease,
                    total_sample=X[self.selected_time_columns[i]].value_counts().to_dict(),
                    criterion=self.criterion,
                    random_state=i + self.random_state,
                )
                for i in range(self.n_estimators)
            )

        return self

    def predict_proba(self, X):
        """
        Predicts probabilities for negative and positive classes.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        if self.multi:
            predictions = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(model.predict)(X) for model in self.n_estimators_
            )
        else:
            predictions = [model.predict(X) for model in self.n_estimators_]

        positive_proba = np.mean(np.array(predictions), axis=0)
        negative_proba = np.ones(len(positive_proba)) - positive_proba

        return np.vstack([negative_proba, positive_proba]).T

    def predict_proba_(self, X):
        return self.predict_proba(X)[:, 1]

    def predict(self, X):
        predictions = self.predict_proba_(X)
        return (predictions >= self.train_target_proportion).astype(int)

    def score(self, X, y):
        predictions = self.predict_proba_(X)
        return metrics.roc_auc_score(y, predictions)

    def feature_importance(self, impurity_decrease=False):
        """
        Retrieves the feature importance as a DataFrame with
        columns ["Feature", "Importance"], aggregated over trees.
        """
        return (
            pd.concat(
                [
                    n_estimator.feature_importance(impurity_decrease=impurity_decrease)
                    for n_estimator in self.n_estimators_
                ]
            )
            .groupby("Feature")
            .sum()
            .sort_values(by="Importance", ascending=False)
        )


class _RandomTimeSplitTree:
    """
    Base tree used inside TimeForestClassifier.
    """

    def __init__(
        self,
        X,
        y,
        row_indexes=[],
        time_column="period",
        max_depth=5,
        max_features="auto",
        bootstrapping=True,
        criterion="gini",
        period_criterion="avg",
        invariance_penalty=10.0,
        min_impurity_decrease=0.0,
        total_sample=None,
        min_sample_periods=100,
        sample_weight=None,
        depth=None,
        verbose=False,
        split_verbose=False,
        impurity_verbose=False,
        random_state=42,
        rng=None,
    ):
        if len(row_indexes) == 0:
            row_indexes = np.arange(len(y))
            X.reset_index(inplace=True, drop=True)
        if depth is None:
            depth = 0
            if bootstrapping:
                resampled_X = X.sample(
                    frac=1.0, replace=True, random_state=random_state
                )
                resampled_idx = resampled_X.index
                X = resampled_X
                X.reset_index(inplace=True, drop=True)
                if isinstance(y, (pd.DataFrame, pd.Series)):
                    y = y.values
                y = y[resampled_idx]

        self.X = X
        self.y = y
        self.row_indexes = row_indexes
        self.max_depth = max_depth
        self.depth = depth
        self.time_column = time_column
        self.min_sample_periods = min_sample_periods
        self.verbose = verbose
        self.split_verbose = split_verbose
        self.impurity_verbose = impurity_verbose
        self.max_features = max_features
        self.split_variable = "LEAF"
        self.bootstrapping = bootstrapping
        self.criterion = criterion
        self.period_criterion = period_criterion
        self.invariance_penalty = invariance_penalty
        self.min_impurity_decrease = min_impurity_decrease
        self.total_sample = total_sample
        self.random_state = random_state
        self.rng = rng if rng is not None else default_rng(self.random_state)

        if sample_weight is not None:
            self.sample_weight = sample_weight
        else:
            self.sample_weight = np.ones(len(y))

        self.n_examples = len(row_indexes)
        self.variables = [col for col in X.columns if col != time_column]
        self.variables = [
            col for col in self.variables if "time_column" not in col
        ]

        if max_features == "auto":
            self.max_n_variables = max(int(len(self.variables) ** 0.5), 1)
        else:
            self.max_n_variables = max(
                int(max_features * len(self.variables)), 1
            )

        self.value = np.mean(y[row_indexes])
        self.score = float("inf")

        if verbose:
            print(f"Depth: {self.depth}")
            print(f"Max Depth: {self.max_depth}")
            print("Node periods distribution")
            print(
                self.X.loc[self.row_indexes, self.time_column]
                .value_counts()
                .sort_index()
            )

        if self.depth == 0:
            if (
                check_min_sample_periods(
                    self.X.loc[self.row_indexes],
                    self.time_column,
                    self.min_sample_periods,
                )
                == 0
            ):
                print(
                    "Not enough sample in the periods to perform a split "
                    + "using {} as minimum sample by period".format(
                        min_sample_periods
                    )
                )

        if self.depth < self.max_depth:
            self.create_split()

    def create_split(self):
        """
        Selects a subset of the input features, searches best split,
        and builds left/right subtrees.
        """
        variables_to_consider = self.rng.choice(
            self.variables, self.max_n_variables, replace=False
        )
        for idx, variable in enumerate(self.variables):
            if variable in variables_to_consider:
                self.find_better_split(variable, idx)
        if self.score == float("inf"):
            return False

        x = self._split_column()
        left_split = np.nonzero(x <= self.split_example)
        right_split = np.nonzero(x > self.split_example)

        self.left_split = _RandomTimeSplitTree(
            self.X,
            self.y,
            self.row_indexes[left_split],
            depth=self.depth + 1,
            max_features=self.max_features,
            bootstrapping=self.bootstrapping,
            min_sample_periods=self.min_sample_periods,
            time_column=self.time_column,
            max_depth=self.max_depth,
            criterion=self.criterion,
            period_criterion=self.period_criterion,
            invariance_penalty=self.invariance_penalty,
            min_impurity_decrease=self.min_impurity_decrease,
            total_sample=self.total_sample,
            sample_weight=self.sample_weight,
            verbose=self.verbose,
            split_verbose=self.split_verbose,
            impurity_verbose=self.impurity_verbose,
            random_state=self.random_state,
            rng=self.rng,
        )
        self.right_split = _RandomTimeSplitTree(
            self.X,
            self.y,
            self.row_indexes[right_split],
            depth=self.depth + 1,
            max_features=self.max_features,
            bootstrapping=self.bootstrapping,
            min_sample_periods=self.min_sample_periods,
            time_column=self.time_column,
            max_depth=self.max_depth,
            criterion=self.criterion,
            period_criterion=self.period_criterion,
            invariance_penalty=self.invariance_penalty,
            min_impurity_decrease=self.min_impurity_decrease,
            total_sample=self.total_sample,
            sample_weight=self.sample_weight,
            verbose=self.verbose,
            split_verbose=self.split_verbose,
            impurity_verbose=self.impurity_verbose,
            random_state=self.random_state,
            rng=self.rng,
        )

    def find_better_split(self, variable, variable_idx):
        """
        Given an input feature variable, it finds the best split possible
        using it. If it is better than the current stored split, it replaces
        it by the current variable and split value.
        """
        x = self.X.loc[self.row_indexes, variable]
        y = self.y[self.row_indexes]
        weights = self.sample_weight[self.row_indexes]

        period_data = self.X.loc[self.row_indexes, self.time_column]
        unique_periods = period_data.unique()
        x = x.values

        sorted_indexes = np.argsort(x)
        sorted_x = x[sorted_indexes]
        sorted_y = y[sorted_indexes]
        sorted_weights = weights[sorted_indexes]
        sorted_period_data = period_data.iloc[sorted_indexes]

        right_periods_count = sorted_period_data.value_counts().to_dict()
        left_periods_count = {key: 0 for key in right_periods_count.keys()}
        right_period_dict = initialize_period_dict(unique_periods)
        left_period_dict = initialize_period_dict(unique_periods)

        right_period_dict = fill_right_dict(
            sorted_period_data.values, sorted_y, sorted_weights, right_period_dict
        )

        for example in range(0, self.n_examples - self.min_sample_periods - 1):
            x_i = sorted_x[example]
            y_i = sorted_y[example]
            period_i = sorted_period_data.iloc[example]
            weight_i = sorted_weights[example]

            right_periods_count[period_i] -= 1
            left_periods_count[period_i] += 1

            right_period_dict[period_i]["count"] -= weight_i
            left_period_dict[period_i]["count"] += weight_i
            right_period_dict[period_i]["sum"] -= y_i * weight_i
            left_period_dict[period_i]["sum"] += y_i * weight_i

            if self.criterion in ("std", "std_norm"):
                right_period_dict[period_i]["squared_sum"] -= (y_i ** 2) * weight_i
                left_period_dict[period_i]["squared_sum"] += (y_i ** 2) * weight_i

            if (
                example < self.min_sample_periods
                or x_i == sorted_x[example + 1]
            ):
                continue
            elif not check_min_sample_periods_dict(
                right_periods_count, self.min_sample_periods
            ) or not check_min_sample_periods_dict(
                left_periods_count, self.min_sample_periods
            ):
                continue
            if not check_min_sample_periods_dict(
                right_periods_count, self.min_sample_periods
            ):
                break

            if self.split_verbose:
                print(f"Evaluate a split on variable {variable} at value {x_i}")

            current_score = score_by_period(
                right_period_dict,
                left_period_dict,
                self.criterion,
                self.period_criterion,
                self.invariance_penalty,
                self.split_verbose,
            )

            if current_score < self.score:
                impurity_decrease = impurity_decrease_by_period(
                    right_period_dict,
                    left_period_dict,
                    self.total_sample,
                    self.period_criterion,
                    self.impurity_verbose,
                )

                if impurity_decrease >= self.min_impurity_decrease:
                    self.split_variable = variable
                    self.score = current_score
                    self.split_example = x_i
                    self.split_variable_idx = variable_idx
                    self.impurity_decrease = impurity_decrease

    def _is_leaf(self):
        return self.score == float("inf")

    def _split_column(self):
        return self.X.values[self.row_indexes, self.split_variable_idx]

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.variables + [self.time_column])
        return np.array([self._predict_row(x) for x in X.iterrows()])

    def _predict_row(self, x):
        if self._is_leaf():
            return self.value
        tree = (
            self.left_split
            if x[1][self.split_variable] <= self.split_example
            else self.right_split
        )
        return tree._predict_row(x)

    def _get_split_variable(self):
        if not self._is_leaf():
            return (
                self.split_variable
                + "@"
                + self.left_split._get_split_variable()
                + "@"
                + self.right_split._get_split_variable()
            )
        return "LEAF"

    def _get_impurity_decrease(self):
        if not self._is_leaf():
            return (
                [self.impurity_decrease]
                + self.left_split._get_impurity_decrease()
                + self.right_split._get_impurity_decrease()
            )
        return ["LEAF"]

    def feature_importance(self, impurity_decrease=False):
        splits = self._get_split_variable()
        splits_features = splits.replace("@LEAF", "").split("@")

        if impurity_decrease:
            impurity_decreases = self._get_impurity_decrease()
            impurity_decreases = [i for i in impurity_decreases if i != "LEAF"]
            importance = impurity_decreases
        else:
            importance = [1 for _ in splits_features]

        return (
            pd.DataFrame(
                zip(splits_features, importance),
                columns=["Feature", "Importance"],
            )
            .groupby("Feature")
            .sum()
        )


def irf_predict_both(
    X_train,
    y_train,
    z_train,
    X_test,
    feature_names=None,
    n_estimators=5,
    max_depth=5,
    min_sample_periods=10,
    invariance_penalty=1e3,
    min_impurity_decrease=0.0,
    random_state=42,
):
    """
    Fit TimeForestClassifier (IRF) once and return:
      - train probabilities (dict key 'train')
      - test probabilities  (dict key 'test')
      - feature importance (dict key 'feature_importance')
    """

    # Ensure numpy arrays
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)
    y_train = np.asarray(y_train, dtype=int).ravel()
    z_train = np.asarray(z_train)

    n_train, n_features = X_train.shape
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(n_features)]

    # Build DataFrame with features + environment column
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train["environment"] = z_train

    # Single-process to avoid joblib pickling issues under reticulate
    model = TimeForestClassifier(
        time_column="environment",
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_sample_periods=min_sample_periods,
        invariance_penalty=invariance_penalty,
        min_impurity_decrease=min_impurity_decrease,
        criterion="gini_invariance_penalty",
        period_criterion="avg",
        n_jobs=1,
        multi=False,
        random_state=random_state,
    )

    # Fit IRF
    model.fit(df_train[feature_names + ["environment"]], y_train, verbose=False)

    # Probabilities for train and test
    train_proba = model.predict_proba(df_train[feature_names])[:, 1]
    df_test     = pd.DataFrame(X_test, columns=feature_names)
    test_proba  = model.predict_proba(df_test[feature_names])[:, 1]

    # Impurity-decrease feature importance
    fi_df = model.feature_importance(impurity_decrease=True)
    # fi_df currently has index = feature, column = 'Importance'
    fi_df = fi_df.reset_index()          # columns: ['Feature', 'Importance']

    fi_dict = {name: 0.0 for name in feature_names}
    for _, row in fi_df.iterrows():
        feat = row["Feature"]
        imp  = float(row["Importance"])
        fi_dict[feat] = imp

    return {
        "train": np.asarray(train_proba, dtype=float),
        "test":  np.asarray(test_proba,  dtype=float),
        "feature_importance": fi_dict,
    }

def anchorboost_predict_both(
    X_train,
    y_train,
    z_train,
    X_test,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    lam=5.0,          # used as gamma. Should be > 1 to enforce invariance.
    random_state=0,
    importance_type="gain",  # "gain" or "split" (LightGBM)
):
    """
    Fit AnchorBooster once on (X_train, y_train, z_train),
    return probabilities for train and test, plus feature importance.

    Parameters
    ----------
    importance_type : {"gain", "split"}
        Measure to use for feature importance, passed to
        lightgbm.Booster.feature_importance(...).
    """

    # --- Preserve feature names if available -------------------------------
    feature_names = None
    if hasattr(X_train, "columns"):
        feature_names = np.asarray(X_train.columns, dtype=object)

    # --- Force all inputs to NumPy arrays ----------------------------------
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)
    y_train = np.asarray(y_train, dtype=int).ravel()

    # Anchor: encode z as integer categories, pass as 1D int array
    z_arr   = np.asarray(z_train)
    z_codes = pd.Categorical(z_arr).codes.astype(np.int64)  # shape (n,)

    booster = AnchorBooster(
        gamma=lam,
        objective="binary",          # binary probit classification
        num_boost_round=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_gain_to_split=0.0,
        seed=random_state,
        verbosity=-1,
    )

    # Fit with anchors Z as 1D int array
    booster.fit(X_train, y_train, Z=z_codes)

    # For objective="binary", predict() returns probabilities in [0,1]
    train_proba = booster.predict(X_train)  # (n_train,)
    test_proba  = booster.predict(X_test)   # (n_test,)

    # --- Feature importance from underlying LightGBM booster ---------------
    lgb_booster = booster.booster_  # lightgbm.Booster
    importances = lgb_booster.feature_importance(
        importance_type=importance_type
    )

    # Wrap in a nice structure (Series) if we have names
    if feature_names is not None and len(feature_names) == importances.shape[0]:
        importance_obj = (
            pd.Series(importances, index=feature_names, name=importance_type)
            .sort_values(ascending=False)
        )
    else:
        # Fall back to plain NumPy array
        importance_obj = np.asarray(importances, dtype=float)

    return {
        "train": np.asarray(train_proba, dtype=float),
        "test":  np.asarray(test_proba,  dtype=float),
        "feature_importance": importance_obj,
        "importance_type": importance_type,
    }
