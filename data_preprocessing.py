import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split function
import numpy as np
import statsmodels.api as sm
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

origin_df = pd.read_csv('customers_copy.csv')

df = origin_df.copy()

unique_locations = df['Location'].unique()
locations_dictionary = {item: index for index, item in enumerate(unique_locations)}
unique_cardTypes = df['CardType'].unique()
cardTypes_dictionary = {item: index for index, item in enumerate(unique_cardTypes)}
unique_genders = df['Gender'].unique()
genders_dictionary = {item: index for index, item, in enumerate(unique_genders)}

df['Location'] = df['Location'].map(locations_dictionary)
df['CardType'] = df['CardType'].map(cardTypes_dictionary)
df['Gender'] = df['Gender'].map(genders_dictionary)

df.to_csv('encoded_data.csv', index=False)

print(genders_dictionary)
print(cardTypes_dictionary)
print(locations_dictionary)


# split dataset in features and target variable
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test
    return x_train, x_test, y_train, y_test


def feature_selection(df, predictor_list):
    feature_cols = predictor_list
    x = df[feature_cols]  # Features
    y = df['AnnualSpending']  # Target variable
    return x, y


def build_regression(x, y):
    x_array = np.asarray(x)
    y_array = np.asarray(y)

    # Check for NaN values
    nan_mask = np.isnan(x_array)
    nan_rows, nan_cols = np.where(nan_mask)
    if np.any(nan_mask):
        print("Rows with NaN values:", nan_rows)
        print("Columns with NaN values:", nan_cols)

    x_array = list(x_array)
    y_array = list(y_array)
    # Fit and make the predictions by the model
    reg = sm.OLS(y_array, x_array).fit()
    predictions = reg.predict(x)
    print(reg.summary())
    return reg


def plot_regression_results(model, x, y):
    # Create a pairplot to visualize the relationships between features and target
    sns.pairplot(data=x.join(y))

    # Plot the predicted values vs. actual values
    y_pred = model.predict(x)
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.show()


def main():
    feature_cols = ['IncomeEstimate', 'Gender', 'Location']
    x, y = feature_selection(df, feature_cols)
    reg = build_regression(x, y)
    plot_regression_results(reg, x, y)


if __name__ == '__main__':
    main()