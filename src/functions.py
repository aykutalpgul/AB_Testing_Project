from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def check_normality(data, group_col, value_col): #normality_test

    """
    Perform Shapiro-Wilk normality test on groups.
    """
    groups = data[group_col].unique()
    results = {}
    for group in groups:
        group_data = data[data[group_col] == group][value_col]
        stat, p = stats.shapiro(group_data)
        results[group] = p
    return results

def check_variance_homogeneity(data, group_col, value_col): #homogenety_test

    """
    Perform Levene's test for variance homogeneity.
    """
    group1 = data[data[group_col] == data[group_col].unique()[0]][value_col]
    group2 = data[data[group_col] == data[group_col].unique()[1]][value_col]
    stat, p = stats.levene(group1, group2)
    return p

def hypothesis_test(data, group_col, value_col):
    """
    Perform independent two-sample t-test or Mann-Whitney U Test depending on assumptions.
    """
    normality = check_normality(data, group_col, value_col)
    variance = check_variance_homogeneity(data, group_col, value_col)
    
    if all(p > 0.05 for p in normality.values()) and variance > 0.05:
        # Parametric
        group1 = data[data[group_col] == data[group_col].unique()[0]][value_col]
        group2 = data[data[group_col] == data[group_col].unique()[1]][value_col]
        stat, p_value = stats.ttest_ind(group1, group2)
        test_used = "Independent T-Test"
    else:
        # Non-parametric
        group1 = data[data[group_col] == data[group_col].unique()[0]][value_col]
        group2 = data[data[group_col] == data[group_col].unique()[1]][value_col]
        stat, p_value = stats.mannwhitneyu(group1, group2)
        test_used = "Mann-Whitney U Test"
    
    return test_used, p_value

def plot_distribution(data, group_col, value_col): #plot_distribution function
    """
    Plot the distribution of values per group.
    """
    sns.histplot(data=data, x=value_col, hue=group_col, kde=True)
    plt.show()

#to call functions copy below as calling a library:
    # from src.functions import check_normality, check_variance_homogeneity, hypothesis_test, plot_distribution