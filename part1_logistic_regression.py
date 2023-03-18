import pandas
import numpy
import clean_data_template
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import kfold_template
import matplotlib.pyplot as pyplot

import pickle

from cryptography.fernet import Fernet
import base64

dataset = pandas.read_csv("survey_dataset.csv")
dataset = clean_data_template.clean_data(dataset)

print(dataset.describe())

'''train the model'''
dataset['game_cat'], uniques = pandas.factorize(dataset.game)
dataset = pandas.get_dummies(dataset, columns = ['gender','region','horoscope'])

target = dataset.game_cat
target = target.values

data = dataset.drop("game_cat", axis = 1)
data = data.drop("game", axis = 1)

feature_list = data.columns
print(len(data.columns))
data = data.values


machine = linear_model.LogisticRegression(multi_class='multinomial',max_iter=20000)
machine.fit(data,target)

feature_importances_raw = machine.coef_
print(feature_importances_raw)

print("\n\n")
print("\n\n")

for i in range(0,7):
    print(feature_importances_raw[i].sum())
    locals()["feature_importances_raw_"+str(i)] = feature_importances_raw[i]/feature_importances_raw[i].sum()
    print(locals()["feature_importances_raw_"+str(i)])
    print(locals()["feature_importances_raw_"+str(i)].sum())

out_arr = feature_importances_raw_0
for i in range(1,7):
    out_arr = numpy.add(locals()["feature_importances_raw_"+str(i)], out_arr)  
print(out_arr)



feature_zip = zip(feature_list, out_arr)
feature_importances = [(feature, round(importance, 4)) for feature,importance in feature_zip]
feature_importances = [(item[0], abs(item[1])) for item in feature_importances]
feature_importances = sorted(feature_importances, key = lambda x: x[1],reverse = True)
[ print('{:13}: {}'.format(*feature_importance)) for feature_importance in feature_importances]


trials = []
for w in range(12,32):  
    feature_importances_sub = feature_importances[:w]
    print([i[0] for i in feature_importances_sub])
    dataset_sl = dataset[[i[0] for i in feature_importances_sub]]
    data = dataset_sl.values
    machine = linear_model.LogisticRegression(multi_class='multinomial',max_iter=20000)
    return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
    return_values = [i[0] for i in return_values]
    average_as=sum(return_values)/len(return_values)
    print(return_values)
    print("Average accuracy score for logistic regression when number of features is ",str(w), ": ",average_as)
    trials.append((w,average_as,[i[0] for i in feature_importances_sub]))
trials.sort(key = lambda x: x[1], reverse = True)
print(trials[:5])

features = trials[0][2]
with open("features.pickle","wb") as file:
    pickle.dump(features, file)


dataset_sl = dataset[features]
print(features)
print(dataset_sl)


