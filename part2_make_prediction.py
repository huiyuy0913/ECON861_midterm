import pandas
import numpy
import clean_data_template
from sklearn import linear_model
from sklearn import svm
import kfold_template
import matplotlib.pyplot as pyplot
import pickle

with open("features.pickle","rb") as file:
	features = pickle.load(file)

print(features)

dataset = pandas.read_csv("survey_dataset.csv")
dataset = clean_data_template.clean_data(dataset)

print(dataset.describe())

dataset['game_cat'], uniques = pandas.factorize(dataset.game)
print(uniques)

number = range(0, len(dataset['game_cat'].unique()))
game_df = pandas.DataFrame(data=number, index=uniques, columns=['game_cat_predict']).reset_index().rename(columns={'index': 'game_predict'})
print(game_df)

dataset = pandas.get_dummies(dataset, columns = ['gender','region','horoscope'])

target = dataset.game_cat
target = target.values

dataset_sl = dataset[features] 
data = dataset_sl.values

machine = svm.SVC(kernel="linear")
machine.fit(data,target)




# make predictions
new_dataset = pandas.read_csv("new_customers_dataset.csv")
print(new_dataset.isna().sum())
print(new_dataset.describe())
# print(new_dataset[new_dataset.personality5.isna()==True])
# print(new_dataset[new_dataset.personality9.isna()==True])

means = new_dataset.mean()
new_dataset_fill = new_dataset.fillna(value=means)
# print(new_dataset.loc[[612]])
# print(new_dataset.loc[[2584]])

new_dataset_fill = pandas.get_dummies(new_dataset_fill, columns = ['gender','region','horoscope'])

new_dataset_sl = new_dataset_fill[features]

print(new_dataset_sl)
new_data_sl = new_dataset_sl.values
new_dataset['game_cat_predict'] = machine.predict(new_data_sl)
print(new_dataset)

merge_new_dataset_sl = new_dataset.merge(game_df, on=['game_cat_predict'], suffixes=['_sl','_df'], how='left')

merge_new_dataset_sl = merge_new_dataset_sl.drop(columns=['game_cat_predict'])
print(merge_new_dataset_sl)

merge_new_dataset_sl.to_csv("new_customers_dataset_with_predictions.csv",index=False)

