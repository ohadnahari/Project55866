import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler


# TODO: write a function to summarize the numbers/statistics of the file
# TODO: delete columns lastname, customerID, unnamed from the dataframe before the processing

origin_df = pd.read_csv('customers_copy.csv')

df = origin_df.copy()


# TODO: the commented code below is redundant with encode_data function
# TODO: it's here for safety measures. Delete after final code execution
# unique_locations = df['Location'].unique()
# locations_dictionary = {item: index for index, item in enumerate(unique_locations)}
# unique_cardTypes = df['CardType'].unique()
# cardTypes_dictionary = {item: index for index, item in enumerate(unique_cardTypes)}
# unique_genders = df['Gender'].unique()
# genders_dictionary = {item: index for index, item, in enumerate(unique_genders)}
#
# # encode the non-numeric values with the integer values assigned
# df['Location'] = df['Location'].map(locations_dictionary)
# df['CardType'] = df['CardType'].map(cardTypes_dictionary)
# df['Gender'] = df['Gender'].map(genders_dictionary)

def encode_data(df, cols: list[str]):
    for col in cols:
        unique_vals = df[col].unique()
        unique_dict = {item: ind for ind, item in enumerate(unique_vals)}
        df[col] = df[col].map(unique_dict)
    return df


cols_to_encode = ['Location', 'CardType', 'Gender']
encoded_df = encode_data(df, cols_to_encode)




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


def turn_to_binary(df, col_name):
    mean = df[col_name].mean()
    new_df = df.copy()
    new_df[col_name] = (new_df[col_name] >= mean).astype(int)

    new_df.to_csv('binary_classification.csv', index=False)
    return new_df


def build_log_regression(lr, x,y):
    lr.fit(x, y)
    print('Regression finished with R^2={0:f}'.format(lr.score(x, y)))


def split_log_regression(dataframe, training_set_frac):
    random_rows = np.random.rand(len(dataframe)) < training_set_frac
    data_train = dataframe[random_rows]
    data_test = dataframe[~random_rows]
    print('Training set size:{0:d}\nTest set size: {1:d}'.format(len(data_train), len(data_test)))
    return data_train, data_test


binary_df = turn_to_binary(df, 'AnnualSpending')


log_data_train, log_data_test = split_log_regression(binary_df, 0.7)


y_train = log_data_train['AnnualSpending']
X_train = log_data_train.drop(columns=["Unnamed: 0", "LastName", "RecordNumber", "CustomerId", 'AnnualSpending'])
y_test = log_data_test['AnnualSpending']
X_test = log_data_test.drop(columns=["Unnamed: 0", "LastName", "RecordNumber", "CustomerId", 'AnnualSpending'])


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(solver='lbfgs', max_iter=10000)
log_model = lr.fit(X_train_scaled, y_train)


predictions = log_model.predict(X_test_scaled)
# Print the size of training and test sets
print('Training set size: {}'.format(len(log_data_train)))
print('Test set size: {}'.format(len(log_data_test)))

# Display a few rows of the test target values
print(y_test.head())

# Evaluate the model
print('Model accuracy on training set: {:.2f}'.format(lr.score(X_train_scaled, y_train)))
print('Model accuracy on test set: {:.2f}'.format(lr.score(X_test_scaled, y_test)))

cm = confusion_matrix(y_test,predictions)
pre = precision_score(y_test,predictions)
rec = recall_score(y_test,predictions)
f1 = f1_score(y_test,predictions)
auc = roc_auc_score(y_test,predictions)

print(f'Precision score: {pre:3.3f}')
print(f'Recall score: {rec:3.3f}')
print(f'F1 score: {f1:3.3f}')
print(f'AUC score: {auc:3.3f}')
print ('Confusion matrix:')
print(cm)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True,fmt="d",annot_kws={"size": 16})# font size
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
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
    print(f"the mean absolute error is: {mean_absolute_error(y_test, y_pred)}")
    print(f"the mean squared error is: {mean_squared_error(y_test, y_pred)}")
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
    # main()
    pass