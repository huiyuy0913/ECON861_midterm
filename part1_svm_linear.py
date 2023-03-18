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

with open("features.pickle","rb") as file:
    features = pickle.load(file)

print(features)
dataset = pandas.read_csv("survey_dataset.csv")
dataset = clean_data_template.clean_data(dataset)

print(dataset.describe())

dataset['game_cat'], uniques = pandas.factorize(dataset.game)
dataset = pandas.get_dummies(dataset, columns = ['gender','region','horoscope'])

target = dataset.game_cat
target = target.values

data = dataset.drop("game_cat", axis = 1)
data = data.drop("game", axis = 1)

feature_list = data.columns
print(len(data.columns))
data = data.values

machine = svm.SVC(kernel="linear")
machine.fit(data,target)

feature_importances_raw = machine.coef_
# print(feature_importances_raw)



for i in range(0,21):
    print(feature_importances_raw[i].sum())
    locals()["feature_importances_raw_"+str(i)] = feature_importances_raw[i]/feature_importances_raw[i].sum()
    print(locals()["feature_importances_raw_"+str(i)])
    print(locals()["feature_importances_raw_"+str(i)].sum())

# print(type(feature_importances_raw_1))
out_arr = feature_importances_raw_0
for i in range(1,21):
    out_arr = numpy.add(locals()["feature_importances_raw_"+str(i)], out_arr)  
# print(out_arr)



feature_zip = zip(feature_list, out_arr)
feature_importances = [(feature, round(importance, 4)) for feature,importance in feature_zip]
# print(feature_importances)
feature_importances = [(item[0], abs(item[1])) for item in feature_importances]
# print(feature_importances)
feature_importances = sorted(feature_importances, key = lambda x: x[1],reverse = True)
[ print('{:13}: {}'.format(*feature_importance)) for feature_importance in feature_importances]
# print(type(out_arr))


trials = []
for w in range(1,32):  # 21 personality1
    feature_importances_sub = feature_importances[:w]
    # print(feature_importances_sub)
    print([i[0] for i in feature_importances_sub])
    dataset_sl = dataset[[i[0] for i in feature_importances_sub]]
    data = dataset_sl.values
    # print(dataset)
    machine = svm.SVC(kernel="linear")
    return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
    return_values = [i[0] for i in return_values]
    average_as=sum(return_values)/len(return_values)
    print(return_values)
    print("Average accuracy score for svm linear when number of features is ",str(w), ": ",average_as)
    trials.append((w,average_as,[i[0] for i in feature_importances_sub]))
trials.sort(key = lambda x: x[1], reverse = True)
print(trials[:5])
dataset_sl = dataset[trials[0][2]]
print(dataset_sl)



return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
return_values = [i[0] for i in return_values]
average_as=sum(return_values)/len(return_values)
print(return_values)
print("Average accuracy score for svm linear when number of features is " + str(len(dataset_sl.columns)) + ": ",average_as)



dataset_sl = dataset_sl.loc[:, ~dataset_sl.columns.str.startswith('horoscope')]
print(dataset_sl.columns)
data = dataset_sl.values
return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
return_values = [i[0] for i in return_values]
average_as=sum(return_values)/len(return_values)
print(return_values)
print("Average accuracy score for svm linear when number of features is " + str(len(dataset_sl.columns)) + ": ",average_as)





# the best one
print("\n\n")
print("choose features from logistic regression's feature importance")
print("\n\n")
dataset_sl = dataset[features]  # mae 0.00698 - 0.00462, as 0.002478 - 0.00103867
print(dataset_sl.columns)
data = dataset_sl.values

mae_train = []
mae_test = []
train_scores, test_scores = list(), list()
machine = svm.SVC(kernel="linear")
for i in range(2,6):
    return_values = kfold_template.run_kfold(data, target, machine, i, False, True, False)
    return_values_test = [i[0] for i in return_values]
    return_values_train = [i[1] for i in return_values]
    mae_train_values = [i[2] for i in return_values]
    mae_test_values = [i[3] for i in return_values]
    train_scores.append(sum(return_values_train)/len(return_values_train))
    test_scores.append(sum(return_values_test)/len(return_values_test))
    mae_train.append(sum(mae_train_values)/len(mae_train_values))
    mae_test.append(sum(mae_test_values)/len(mae_test_values))
    print(return_values)
    print("Average accuracy score for svm linear test: ",sum(return_values_test)/len(return_values_test))
    print("Average accuracy score for svm linear train: ",sum(return_values_train)/len(return_values_train))
    print("mean_absolute_error for svm linear train: ",sum(mae_train_values)/len(mae_train_values))
    print("mean_absolute_error for svm linear test: ",sum(mae_test_values)/len(mae_test_values))

values = [i for i in range(2,6)]
pyplot.plot(values, mae_train, '-o', label='Train')  # 0.009159 -0.00675
pyplot.plot(values, mae_test, '-o', label='Test')
pyplot.legend()
pyplot.title("Mean Absolute Error SVM Linear Less Features")
pyplot.savefig("mae_svm_linear_fold_less_feature.png")  
pyplot.show()
pyplot.close()

values = [i for i in range(2,6)]
pyplot.plot(values, train_scores, '-o', label='Train') # 0.003589 - 0.00268
pyplot.plot(values, test_scores, '-o', label='Test') 
pyplot.legend()
pyplot.title("Accuracy Score SVM Linear Less Features")
pyplot.savefig("as_svm_linear_fold_less_feature.png")
pyplot.show()











