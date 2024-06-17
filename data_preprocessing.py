import pandas as pd
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