import pandas as pd

# the first assignment is: Data Exploration: a. Load the dataset and perform an initial exploration to understand its
# structure, data types, and summary statistics. b. Identify any missing values, outliers, or inconsistencies in the
# data. Apply appropriate preprocessing techniques to handle these issues.

#  for inconsistency, we can use the following code to check the data type of each column.
# also we can make sure that LoyaltyYears is smaller than Age

# for missing values, we can use the following code to check the missing values in the dataset
# we will fill the missing values with the mean of the column (for Age and LoyaltyYears, and more)


def open_data():
    df = pd.read_csv('./customers_annual_spending_dataset.csv')
    return df


def all_columns_names(df):
    column_names = df.columns.tolist()
    return column_names


def check_data_type(df):
    print(df.dtypes)


def find_missing_values(df):
    # column_names = df.columns.tolist()
    # for column in column_names:
    #
    columns_with_nulls = df.columns[df.isnull().any()]
    df_columns_with_nulls = df[columns_with_nulls]
    print(df_columns_with_nulls)


def is_loyalty_years_smaller_than_age(df):
    # check if LoyaltyYears is smaller than Age
    loyalty_and_age = df[df['LoyaltyYears'].values < df['Age'].values]
    broken_loyalty_and_age = df[df['LoyaltyYears'].values >= df['Age'].values]
    return loyalty_and_age, broken_loyalty_and_age


def find_mean_of_column(df, column_name):
    mean = df[column_name].mean()
    return mean


def find_median_of_column(df, column_name):
    median = df[column_name].median()
    return median


def find_mode_of_column(df, column_name):
    """
    find the mode of the column - the value that appears the most
    :param df:
    :param column_name:
    :return:
    """
    mode = df[column_name].mode()
    return mode


def create_dict_of_mean_median_mode(df):
    column_names = df.columns.tolist()
    dict_of_mean_median_mode = {}
    for column in column_names:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            mean = find_mean_of_column(df, column)
            median = find_median_of_column(df, column)
            mode = find_mode_of_column(df, column)
            dict_of_mean_median_mode[column] = {'mean': mean, 'median': median, 'mode': mode}
    return dict_of_mean_median_mode


def main():
    df = open_data()
    # print(create_dict_of_mean_median_mode(df))
    # for column in df.columns:
    #     if df[column].dtype == 'float64' or df[column].dtype == 'int64':
    #
    #         # print(column)
    #         # print(df[column].describe())
    #         print(find_mean_of_column(df, column))
    # print(all_columns_names(df))
    check_data_type(df)
    # print(is_loyalty_years_smaller_than_age(df))
    # print(df.info)


if __name__ == "__main__":
    main()
