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


def find_missing_values(df):
    missing_values = df.isnull().sum()
    print(missing_values)


def main():
    df = open_data()
    print(df.info)


if __name__ == "__main__":
    main()
