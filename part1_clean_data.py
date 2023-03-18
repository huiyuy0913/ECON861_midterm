import pandas
import numpy

dataset = pandas.read_csv("survey_dataset.csv")
print(dataset)


'''find the problematic observations''' 
print("\n\n")
print("Since there are some NaN values in the data, I drop all rows with NaN values and reset the index. ")
print("\n\n")

print(dataset.isna().sum())
# print(dataset[dataset.personality5.isna()==True])
dataset = dataset.dropna().reset_index(drop=True)
# print(dataset.isna().sum())
# print(dataset)

print("\n\n")
print("age couldn't be negative, drop all abservations that age is less than 0")
print("\n\n")
print(dataset.describe())
# print(dataset.personality2.describe())
# print(len(dataset[dataset["age"]<0].age))

dataset = dataset[dataset['age']>=0]
# print(dataset.describe())
print("\n\n")
print("check gender, region, horoscope, game")
print("\n\n")
print(dataset.gender.nunique())
print(dataset.gender.value_counts())

print(dataset.region.nunique())
print(dataset.region.value_counts())

print(dataset.horoscope.nunique())
print(dataset.horoscope.value_counts())

print(dataset.game.nunique())
print(dataset.game.value_counts())


print("\n\n")
print("drop the problematic observations in personality2, personality4 and personality5")
print("\n\n")
numeric_col = pandas.to_numeric(dataset['personality2'], errors='coerce')
non_numeric_rows = dataset[numeric_col.isna()]
# print("the problematic observation in personality2")
print(non_numeric_rows)
numeric_col = pandas.to_numeric(dataset['personality4'], errors='coerce')
non_numeric_rows = dataset[numeric_col.isna()]
# print("the problematic observation in personality4")
print(non_numeric_rows)
numeric_col = pandas.to_numeric(dataset['personality5'], errors='coerce')
non_numeric_rows = dataset[numeric_col.isna()]
# print("the problematic observation in personality5")
print(non_numeric_rows)

# dataset['personality2'] = dataset['personality2'].replace('I.1779', '1.1779').astype(float)
# dataset['personality2'] = dataset['personality2'].str.replace('I', '1').astype(float)
replace_dict_2 = {'I': 1}
replace_dict_45 = {'O': 0}
dataset['personality2'] = dataset['personality2'].replace(replace_dict_2, regex=True).astype(float)
dataset['personality4'] = dataset['personality4'].replace(replace_dict_45, regex=True).astype(float)
dataset['personality5'] = dataset['personality5'].replace(replace_dict_45, regex=True).astype(float)
print(dataset.describe())
# print(dataset[dataset.personality4.isna()==True])


print("\n\n")
