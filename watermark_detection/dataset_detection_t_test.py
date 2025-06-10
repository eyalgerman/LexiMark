import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from scipy import stats

SIGNIFICANCE_LEVEL = 0.05

def prepare_data_for_ttest(file_path, score_column, label_column="label", samples_per_class=None):
    data = pd.read_csv(file_path)

    # Filter and sample data
    def sample_data(group):
        if samples_per_class is not None:
            return group.sample(n=samples_per_class, random_state=1)  # fixed seed for reproducibility
        return group

    # Group by label and apply sampling
    sampled_data = data.groupby(label_column).apply(sample_data).reset_index(drop=True)

    # Check if the sampled data for each class is empty
    if sampled_data[sampled_data[label_column] == 1].empty or sampled_data[sampled_data[label_column] == 0].empty:
        return None, "One of the groups is empty, cannot perform T-test."

    # Extract scores and remove NaNs
    scores_label_1 = sampled_data[sampled_data[label_column] == 1][score_column].dropna()
    scores_label_0 = sampled_data[sampled_data[label_column] == 0][score_column].dropna()

    return (scores_label_1, scores_label_0), None


def perform_sampled_t_test(file_path, score_column, label_column="label", samples_per_class=None, repeats=1):
    # Load data
    data = pd.read_csv(file_path)

    # Define containers for p-values and t-stats
    p_values = []
    t_stats = []

    for _ in range(repeats):
        # Filter and sample data
        def sample_data(group):
            # Return entire group if samples_per_class is None or if the group size is smaller than the requested samples
            if samples_per_class is None or len(group) < samples_per_class:
                return group
            return group.sample(n=samples_per_class, random_state=np.random.randint(low=0, high=10000))  # dynamic seed for each repeat

        # Group by label and apply sampling
        sampled_data = data.groupby(label_column).apply(sample_data).reset_index(drop=True)

        # Check if the sampled data for each class is empty
        if sampled_data[sampled_data[label_column] == 1].empty or sampled_data[sampled_data[label_column] == 0].empty:
            continue

        # Extract scores and remove NaNs
        scores_label_1 = sampled_data[sampled_data[label_column] == 1][score_column].dropna()
        scores_label_0 = sampled_data[sampled_data[label_column] == 0][score_column].dropna()

        # Check for sufficient data after dropping NaN values
        if scores_label_1.empty or scores_label_0.empty:
            continue

        # Perform the T-test
        t_stat, p_value = stats.ttest_ind(scores_label_1, scores_label_0, equal_var=False)
        p_values.append(p_value)
        t_stats.append(t_stat)

    # Calculate average p-value and print results
    if p_values:
        avg_p_value = np.mean(p_values)
        significance_level = SIGNIFICANCE_LEVEL  # Adjusted significance level if necessary
        print(f"Average P-value: {avg_p_value:.2e}")
        avg_t_stat = np.mean(t_stats)
        print(f"Average T-statistic: {avg_t_stat:.2f}")
        if avg_p_value < significance_level:
            print("Overall, the test results are statistically significant.")
        else:
            print("Overall, the test results are not statistically significant.")
    else:
        print("No valid tests were performed.")

    return avg_p_value if p_values else None


def perform_t_test_within_file(file_path, score_column, label_column="label"):
    data = pd.read_csv(file_path)

    # Filter data and ensure it is not empty
    scores_label_1 = data[data[label_column] == 1][score_column]
    scores_label_0 = data[data[label_column] == 0][score_column]

    if scores_label_1.empty or scores_label_0.empty:
        return "One of the groups is empty, cannot perform T-test."

    # Check for NaN values and drop them if necessary
    scores_label_1 = scores_label_1.dropna()
    scores_label_0 = scores_label_0.dropna()

    # Perform the T-test
    t_stat, p_value = stats.ttest_ind(scores_label_1, scores_label_0, equal_var=False)

    # Generate textual analysis
    significance_level = SIGNIFICANCE_LEVEL  # Commonly used significance level
    print(f"T-statistic: {t_stat:.2f}")
    print(f"P-value: {p_value:.2e}")

    if p_value < significance_level:
        print("The test results are statistically significant. There is a significant difference between the two groups.")
    else:
        print("The test results are not statistically significant. There is no significant difference between the two groups.")

    # Return the numerical results too if needed elsewhere
    return t_stat, p_value


def perform_t_test_scores(scores_label_to_test, scores_label_non_member):
    """
    Perform a T-test between two sets of scores.
    Args:
        scores_label_to_test (pd.Series): Scores for the label to test (e.g., label 1).
        scores_label_non_member (pd.Series): Scores for the non-member label (e.g., label 0).
    Returns:
        tuple: T-statistic, p-value, and member status (1 if significant, 0 otherwise).
    """
    # Ensure both series are not empty
    if scores_label_to_test.empty or scores_label_non_member.empty:
        return None, "One of the groups is empty, cannot perform T-test."

    # Check for NaN values and drop them if necessary
    scores_label_to_test = scores_label_to_test.dropna()
    scores_label_non_member = scores_label_non_member.dropna()

    # Perform the T-test
    t_stat, p_value = stats.ttest_ind(scores_label_to_test, scores_label_non_member, equal_var=False)

    member = 1 if p_value < SIGNIFICANCE_LEVEL else 0

    return t_stat, p_value, member