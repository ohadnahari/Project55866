import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split function

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
    y = df.win_category  # Target variable
    return x, y