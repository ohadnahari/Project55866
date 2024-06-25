import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split function
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from scipy.stats.stats import pearsonr


origin_df = pd.read_csv('customers_copy.csv')

df = origin_df.copy()

# assign integer values to non-numerical values using enumerate
unique_locations = df['Location'].unique()
locations_dictionary = {item: index for index, item in enumerate(unique_locations)}
unique_cardTypes = df['CardType'].unique()
cardTypes_dictionary = {item: index for index, item in enumerate(unique_cardTypes)}
unique_genders = df['Gender'].unique()
genders_dictionary = {item: index for index, item, in enumerate(unique_genders)}

# encode the non-numeric values with the integer values assigned
df['Location'] = df['Location'].map(locations_dictionary)
df['CardType'] = df['CardType'].map(cardTypes_dictionary)
df['Gender'] = df['Gender'].map(genders_dictionary)

rows_to_delete = []
for index, row in df.iterrows():
    if row['AnnualSpending'] < 0:
        rows_to_delete.append(index)

df.drop(index=rows_to_delete, inplace=True)


# save the encoded data
df.to_csv('encoded_data.csv', index=False)

# split dataset in features and target variable
def split_data(x, y):
    """
    Splits data into train and test sets
    """
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
    training_set = pd.read_csv('encoded_data.csv')
    training_set = training_set.drop(columns=["Unnamed: 0", "LastName", "RecordNumber", "CustomerId"])
    # print(training_set.columns)
    X = training_set.drop(columns=["AnnualSpending"])
    y = training_set["AnnualSpending"]

    x_train, x_test, y_train, y_test = split_data(X, y)
    lm = LinearRegression().fit(x_train, y_train)
    y_pred = lm.predict(x_test)
    # print(f"the mean absolute error is: {mean_absolute_error(y_test, y_pred)}")
    # print(f"the mean squared error is: {mean_squared_error(y_test, y_pred)}")
    formula = 'AnnualSpending ~ ' + '+'.join(X.columns)
    smf_model = smf.ols(formula=formula, data=training_set)
    results = smf_model.fit()
    # print(formula)
    # print(results.summary())

    training_set_fraction = 0.6
    msk = np.random.rand(len(training_set)) < training_set_fraction
    data_train = training_set[msk]
    data_test = training_set[~msk]
    # print('Training set size: {0:d}\nTest set size: {1:d}'.format(len(data_train), len(data_test)))

    y_predicted = results.predict(data_test)
    plt.scatter(data_test['AnnualSpending'], y_predicted)
    plt.xlabel('real data'), plt.ylabel('predicted data')

    # Adding the line y=x
    lims = [np.min([data_test['AnnualSpending'].min(), y_predicted.min()]),
            np.max([data_test['AnnualSpending'].max(), y_predicted.max()])]
    plt.plot(lims, lims, 'r--', alpha=0.75, label='y = x')

    # Adding the regression line
    # Calculating the regression line
    regression_slope, regression_intercept = np.polyfit(data_test['AnnualSpending'], y_predicted, 1)
    regression_line = regression_slope * np.array(lims) + regression_intercept
    plt.plot(lims, regression_line, 'b-', alpha=0.75, label='Regression Line')

    # Adding a legend
    plt.legend()

    # Showing the plot
    plt.show()

    # feature_cols = ['IncomeEstimate', 'Gender', 'Location']
    # x, y = feature_selection(df, feature_cols)
    # reg = build_regression(x, y)
    # plot_regression_results(reg, x, y)


if __name__ == '__main__':
    main()