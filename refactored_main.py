import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import statistics
import pprint


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 1 - Exploratory functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def check_data_type(dataframe):
    print(dataframe.dtypes)


def find_missing_values(dataframe):
    columns_with_nulls = dataframe.columns[dataframe.isnull().any()]
    df_columns_with_nulls = dataframe[columns_with_nulls]
    print(df_columns_with_nulls)


def find_mean_of_column(dataframe, column_name):
    mean = dataframe[column_name].mean()
    return mean


def find_median_of_column(dataframe, column_name):
    median = dataframe[column_name].median()
    return median


def find_mode_of_column(dataframe, column_name):
    """
    find the mode of the column - the value that appears the most
    :param dataframe:
    :param column_name:
    :return:
    """
    mode = dataframe[column_name].mode()[0]
    return mode


def statistics_dictionary(dataframe):
    """
    create a dictionary of the mean, median, and mode of each column
    :param dataframe:
    :return:
    """
    column_names = dataframe.columns.tolist()
    dict_of_mean_median_mode = {}
    for column in column_names:
        if dataframe[column].dtype == 'float64' or dataframe[column].dtype == 'int64':
            if column in ['RecordNumber', 'CustomerId']:
                continue
            mean = find_mean_of_column(dataframe, column)
            median = find_median_of_column(dataframe, column)
            mode = find_mode_of_column(dataframe, column)
            std = statistics.stdev(dataframe[column])
            cv = round(std / mean, 2)
            min_obs = dataframe[column].min()
            max_obs = dataframe[column].max()
            observations = len(dataframe[column])
            unique_count = len(dataframe[column].unique())
            dict_of_mean_median_mode[column] = {'mean': mean,
                                                'median': median,
                                                'mode': mode,
                                                'std': std,
                                                'cv': cv,
                                                'min': min_obs,
                                                'max': max_obs,
                                                'observations': observations,
                                                'unique_count': unique_count}
        else:
            mode = find_mode_of_column(dataframe, column)
            dict_of_mean_median_mode[column] = {'mode': mode}
    return dict_of_mean_median_mode


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 2 - Data Cleaning and Preparation Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def is_loyalty_years_smaller_than_age(dataframe):
    """
    find the rows where the loyalty years are greater than the age
    :param dataframe:
    :return: list of record numbers where the loyalty years are greater than the age
    """
    broken_loyalty_and_age = dataframe[dataframe['LoyaltyYears'].values > dataframe['Age'].values]
    lst = [val for val in broken_loyalty_and_age['RecordNumber']]
    return lst


def positive_column(dataframe, column_name):
    """
    find the rows where the column has negative values
    :param dataframe:
    :param column_name:
    :return: true if there are no negative values, otherwise a list of record numbers where the column has negative values
    """
    vals = dataframe[column_name].values
    negative_indices = np.where(vals < 0)[0]
    if negative_indices.size > 0:
        lst = [val for val in negative_indices]
        return lst
    return True


def fill_missing_numerical_values(dataframe):
    """
    fill the missing values with the mean of the column
    :param dataframe:
    :return:
    """
    column_names = dataframe.columns.tolist()
    for column in column_names:
        if dataframe[column].dtype == 'float64' or dataframe[column].dtype == 'int64':
            mean = round(find_mean_of_column(dataframe, column), 0)
            dataframe[column].fillna(mean, inplace=True)
    return dataframe


def fill_missing_categorical_values(dataframe):
    """
    fill the missing values with the mode of the column
    :param dataframe:
    :return:
    """
    column_names = dataframe.columns.tolist()
    for column in column_names:
        if dataframe[column].dtype == 'object':
            mode = find_mode_of_column(dataframe, column)
            dataframe[column].fillna(mode, inplace=True)
    return dataframe


def fill_all_missing_values(dataframe):
    """
    fill all missing values in the dataset
    :param dataframe:
    :return:
    """
    fill_missing_numerical_values(dataframe)
    fill_missing_categorical_values(dataframe)
    return dataframe


def encode_data(dataframe, cols: list[str]):
    """
    encode the categorical columns
    :param dataframe:
    :param cols:
    :return:
    """
    for col in cols:
        unique_vals = dataframe[col].unique()
        unique_dict = {item: ind for ind, item in enumerate(unique_vals)}
        dataframe[col] = dataframe[col].map(unique_dict)
    return dataframe


def del_rows(dataframe, row_indices: list[int]):
    """
    delete the rows from the dataframe
    :param dataframe:
    :param row_indices:
    :return:
    """
    dataframe = dataframe.drop(row_indices, inplace=True)
    return dataframe


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 3 - Linear Regression
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def perform_linear_regression(dataframe, training_set_fraction=0.7):
    """
    perform linear regression on the dataset
    :param dataframe:
    :param training_set_fraction:
    :return: plot with regression line and y=x line
    """
    dataframe = dataframe.drop(columns=["LastName", "RecordNumber", "CustomerId"])
    x = dataframe.drop(columns=["AnnualSpending"])
    y = dataframe["AnnualSpending"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=training_set_fraction, random_state=1)

    # scikit-learn model
    lm = LinearRegression().fit(x_train, y_train)
    y_pred = lm.predict(x_test)

    print(f"the mean absolute error is: {mean_absolute_error(y_test, y_pred)}")
    print(f"the mean squared error is: {mean_squared_error(y_test, y_pred)}")

    # statsmodels model
    formula = 'AnnualSpending ~ ' + '+'.join(x.columns)
    smf_model = smf.ols(formula=formula, data=dataframe)
    results = smf_model.fit()
    y_predicted = results.predict(x_test)

    # Show the summary of the statsmodels linear regression
    print(results.summary())

    # Plot
    plt.scatter(y_test, y_predicted)
    plt.xlabel('actual data'), plt.ylabel('predicted data')

    # Adding the line y=x
    lims = [np.min([y_test.min(), y_predicted.min()]),
            np.max([y_test.max(), y_predicted.max()])]
    plt.plot(lims, lims, 'r--', alpha=0.75, label='y = x')

    # Adding the regression line
    regression_slope, regression_intercept = np.polyfit(y_test, y_predicted, 1)
    regression_line = regression_slope * np.array(lims) + regression_intercept
    plt.plot(lims, regression_line, 'b-', alpha=0.75, label='Regression Line')

    # Adding a legend, title and showing the plot
    plt.legend()
    plt.title("Linear Regression Model Predicting Annual Spending")
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Part 4 - Logistic Regression
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def turn_col_to_binary(dataframe, col_name):
    """
    turn the column into a binary column, depending on whether the value is greater than the mean
    :param dataframe:
    :param col_name:
    :return:
    """
    new_df = dataframe.copy()
    mean = dataframe[col_name].mean()
    new_df[col_name] = (new_df[col_name] >= mean).astype(int)
    return new_df


def split_log_regression(dataframe, training_set_frac):
    """
    split the data into training and test sets
    :param dataframe:
    :param training_set_frac:
    :return:
    """
    random_rows = np.random.rand(len(dataframe)) < training_set_frac
    data_train = dataframe[random_rows]
    data_test = dataframe[~random_rows]
    return data_train, data_test


def initiate_log_regression(train_data, test_data):
    """
    perform logistic regression on the dataset, print evaluation metrics, show confusion matrix
    :param train_data:
    :param test_data:
    :return:
    """
    # part 1 - perform logistic regression
    lr = LogisticRegression(solver='lbfgs', max_iter=10000)
    scaler = StandardScaler()

    y_train = train_data['AnnualSpending']
    x_train = train_data.drop(columns=["LastName", "RecordNumber", "CustomerId", 'AnnualSpending'])
    y_test = test_data['AnnualSpending']
    x_test = test_data.drop(columns=["LastName", "RecordNumber", "CustomerId", 'AnnualSpending'])

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    log_model = lr.fit(x_train_scaled, y_train)
    predictions = log_model.predict(x_test_scaled)

    # part 2 - evaluate the model
    print('Model accuracy on training set: {:.2f}'.format(lr.score(x_train_scaled, y_train)))
    print('Model accuracy on test set: {:.2f}'.format(lr.score(x_test_scaled, y_test)))

    cm = confusion_matrix(y_test, predictions)
    pre = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)

    print(f'Precision score: {pre:3.3f}')
    print(f'Recall score: {rec:3.3f}')
    print(f'F1 score: {f1:3.3f}')
    print(f'AUC score: {auc:3.3f}')
    print('Confusion matrix:')
    print(cm)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 16})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("Confusion Matrix for Logistic Regression Model")
    plt.show()


def main():
    """
    main function to run all of the above
    :return:
    """
    df = pd.read_csv("customers_annual_spending_dataset.csv").copy()

    # phase 1 - data validation
    loyalty_larger_than_age = is_loyalty_years_smaller_than_age(df)
    negative_spending = positive_column(df, 'AnnualSpending')
    rows_to_delete = loyalty_larger_than_age + negative_spending
    del_rows(df, rows_to_delete)

    # phase 2 - data preparation
    df_without_blanks = fill_all_missing_values(df)
    cols_to_encode = ['Location', 'CardType', 'Gender']
    print('Dataframe Statistics:\n')
    pprint.pprint(statistics_dictionary(df_without_blanks), indent=4, width=40)
    encoded_df = encode_data(df_without_blanks, cols_to_encode)
    encoded_df.to_csv('refactored_output.csv', index=False)

    # phase 3 - linear regression
    print("\nLinear Regression!:\n")
    perform_linear_regression(encoded_df)

    # phase 4 - logistics regression
    print("\nLogistic Regression!:\n")
    binary_tree_df = turn_col_to_binary(encoded_df, 'AnnualSpending')
    log_data_train, log_data_test = split_log_regression(binary_tree_df, 0.7)
    initiate_log_regression(log_data_train, log_data_test)


if __name__ == '__main__':
    main()
