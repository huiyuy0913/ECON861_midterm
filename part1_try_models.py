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


dataset = pandas.read_csv("survey_dataset.csv")
dataset = clean_data_template.clean_data(dataset)

print(dataset.describe())

'''get dummy and factorize'''
print("\n\n")
print("get dummies and factorize")
print("\n\n")
dataset['game_cat'], uniques = pandas.factorize(dataset.game)
# print(uniques)
dataset = pandas.get_dummies(dataset, columns = ['gender','region','horoscope'])

target = dataset.game_cat
target = target.values

data = dataset.drop("game_cat", axis = 1)
data = data.drop("game", axis = 1)

feature_list = data.columns
data = data.values


'''try different models before choose different set of parameters'''
print("\n\n")
print("try different models before choose different set of parameters")
print("\n\n")

# try knc  0.78
print("\n\n")
print("try knc")
print("\n\n")
trials = []
mae_train = []
mae_test = []
train_scores, test_scores = list(), list()

for w in ['uniform', 'distance']:
	for k in range(2,30):  # 15 - 30, 2,50
		machine = KNeighborsClassifier(n_neighbors = k, weights = w)
		return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
		# print(return_values)
		return_values_test = [i[0] for i in return_values]
		return_values_train = [i[1] for i in return_values]
		mae_train_values = [i[2] for i in return_values]
		mae_test_values = [i[3] for i in return_values]

		# all_r2 = [i[0] for i in return_values]
		# print("Average R2: ",sum(all_r2)/len(all_r2))
		average_as = sum(return_values_test)/len(return_values_test)
		trials.append((average_as,k,w))
		mae_train.append(sum(mae_train_values)/len(mae_train_values))
		mae_test.append(sum(mae_test_values)/len(mae_test_values))
		train_scores.append(sum(return_values_train)/len(return_values_train))
		test_scores.append(sum(return_values_test)/len(return_values_test))

	if w == 'uniform':
		mae_train = mae_train
	elif w == 'distance':
		mae_train = mae_train[-28:]
		mae_test = mae_test[-28:]
		train_scores = train_scores[-28:]
		test_scores = test_scores[-28:]

	# print(trials)
	values = [i for i in range(2, 30)]
	pyplot.plot(values, mae_train, '-o', label='Train') # uniform:0.04 - 0.02, distance:>0.4 overfitting
	pyplot.plot(values, mae_test, '-o', label='Test')
	pyplot.legend()
	pyplot.title("Mean Absolute Error knc")
	pyplot.savefig("mae_knc_neignbor_"+str(w)+".png")
	pyplot.show()
	pyplot.close()

	values = [i for i in range(2, 30)]
	pyplot.plot(values, train_scores, '-o', label='Train')
	pyplot.plot(values, test_scores, '-o', label='Test') #uniform:0.025 - 0.02, distance:1 vs 0.78 overfitting
	pyplot.legend()
	pyplot.title("Accuracy Score knc")
	pyplot.savefig("as_knc_neignbor_"+str(w)+".png")
	pyplot.show()


trials.sort(key = lambda x: x[0], reverse = True)
print("\n\n")
print("the accuracy score for knc is")
print(trials[:5])




# try logistic model
print("\n\n")
print("try logistic model")
print("\n\n")
mae_train = []
mae_test = []
train_scores, test_scores = list(), list()
machine = linear_model.LogisticRegression(multi_class='multinomial',max_iter=20000) 
for i in range(2,6):

	return_values = kfold_template.run_kfold(data, target, machine, i, False, True, False)
	return_values_test = [i[0] for i in return_values]
	return_values_train = [i[1] for i in return_values]
	mae_train_values = [i[2] for i in return_values]
	mae_test_values = [i[3] for i in return_values]

	mae_train.append(sum(mae_train_values)/len(mae_train_values))
	mae_test.append(sum(mae_test_values)/len(mae_test_values))
	train_scores.append(sum(return_values_train)/len(return_values_train))
	test_scores.append(sum(return_values_test)/len(return_values_test))


	print(return_values)
	print("Average accuracy score for logistic model test: ",sum(return_values_test)/len(return_values_test))
	print("Average accuracy score for logistic model train: ",sum(return_values_train)/len(return_values_train))
	print("mean_absolute_error for logistic model test: ",sum(mae_test_values)/len(mae_test_values))
	print("mean_absolute_error for logistic model train: ",sum(mae_train_values)/len(mae_train_values))

values = [i for i in range(2,6)]
pyplot.plot(values, mae_train, '-o', label='Train') # 0.012-0.004
pyplot.plot(values, mae_test, '-o', label='Test')
pyplot.legend()
pyplot.title("Mean Absolute Error Logistic")
pyplot.savefig("mae_logistic_fold.png")
pyplot.show()
pyplot.close()

values = [i for i in range(2,6)]
pyplot.plot(values, train_scores, '-o', label='Train') # 0.005 - 0.003
pyplot.plot(values, test_scores, '-o', label='Test') 
pyplot.legend()
pyplot.title("Accuracy Score Logistic")
pyplot.savefig("as_logistic_fold.png")
pyplot.show()



# try svm model 
print("\n\n")
print("try SVM linear")
print("\n\n")
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
	print("mean_absolute_error for svm linear test: ",sum(mae_test_values)/len(mae_test_values))
	print("mean_absolute_error for svm linear train: ",sum(mae_train_values)/len(mae_train_values))
values = [i for i in range(2,6)]
pyplot.plot(values, mae_train, '-o', label='Train')  # 0.017 - 0.02
pyplot.plot(values, mae_test, '-o', label='Test')
pyplot.legend()
pyplot.title("Mean Absolute Error SVM linear")
pyplot.savefig("mae_svm_linear_fold.png")
pyplot.show()
pyplot.close()

values = [i for i in range(2,6)]
pyplot.plot(values, train_scores, '-o', label='Train') # 0.007 - 0.006
pyplot.plot(values, test_scores, '-o', label='Test') 
pyplot.legend()
pyplot.title("Accuracy Score SVM linear")
pyplot.savefig("as_svm_linear_fold.png")
pyplot.show()



print("\n\n")
print("try SVM poly")
print("\n\n")
mae_train = []
mae_test = []
train_scores, test_scores = list(), list()
machine = svm.SVC(kernel="poly") # 0.79
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
	print("Average accuracy score for svm poly test: ",sum(return_values_test)/len(return_values_test))
	print("Average accuracy score for svm poly train: ",sum(return_values_train)/len(return_values_train))
	print("mean_absolute_error for svm poly test: ",sum(mae_test_values)/len(mae_test_values))
	print("mean_absolute_error for svm poly train: ",sum(mae_train_values)/len(mae_train_values))
values = [i for i in range(2,6)]
pyplot.plot(values, mae_train, '-o', label='Train') # 0.00339 - 0.00197
pyplot.plot(values, mae_test, '-o', label='Test')
pyplot.legend()
pyplot.title("Mean Absolute Error SVM poly")
pyplot.savefig("mae_svm_poly_fold.png")
pyplot.show()

values = [i for i in range(2,6)]
pyplot.plot(values, train_scores, '-o', label='Train') # 0.00179 - 0.001058
pyplot.plot(values, test_scores, '-o', label='Test') 
pyplot.legend()
pyplot.title("Accuracy Score SVM poly")
pyplot.savefig("as_svm_poly_fold.png")
pyplot.show()





# try decision tree not this one 0.68
print("\n\n")
print("try decision tree")
print("\n\n")
mae_train = []
mae_test = []
train_scores, test_scores = list(), list()
for i in range(1,20):
	machine = tree.DecisionTreeClassifier(criterion = "gini", max_depth = i)  
	return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
	return_values_test = [i[0] for i in return_values]
	return_values_train = [i[1] for i in return_values]
	mae_train_values = [i[2] for i in return_values]
	mae_test_values = [i[3] for i in return_values]
	print(return_values)
	print("Average accuracy score for decision tree test when depth is "+str(i)+": ",sum(return_values_test)/len(return_values_test))
	print("Average accuracy score for decision tree train when depth is "+str(i)+": ",sum(return_values_train)/len(return_values_train))
	print("mean_absolute_error for decision tree test when depth is "+str(i)+": ",sum(mae_test_values)/len(mae_test_values))
	print("mean_absolute_error for decision tree train when depth is "+str(i)+": ",sum(mae_train_values)/len(mae_train_values))
	train_scores.append(sum(return_values_train)/len(return_values_train))
	test_scores.append(sum(return_values_test)/len(return_values_test))
	mae_train.append(sum(mae_train_values)/len(mae_train_values))
	mae_test.append(sum(mae_test_values)/len(mae_test_values))

values = [i for i in range(1, 20)]
pyplot.plot(values, mae_train, '-o', label='Train') # 
pyplot.plot(values, mae_test, '-o', label='Test')
pyplot.legend()
pyplot.title("Mean Absolute Error Decision Tree")
pyplot.savefig("mae_decision_tree_depth.png")
pyplot.show()
pyplot.close()

values = [i for i in range(1, 20)]
pyplot.plot(values, train_scores, '-o', label='Train') #
pyplot.plot(values, test_scores, '-o', label='Test') 
pyplot.legend()
pyplot.title("Accuracy Score Decision Tree")
pyplot.savefig("as_decision_tree_depth.png")
pyplot.show()



# try random forest overfitting!!!!
print("\n\n")
print("try random forest")
print("\n\n")
mae_train = []
mae_test = []
train_scores, test_scores = list(), list()
for i in range(1,21):
	machine = RandomForestClassifier(criterion = "gini", max_depth = i, n_estimators=100, bootstrap = True)
	return_values = kfold_template.run_kfold(data, target, machine, 4, False, True, False)
	return_values_test = [i[0] for i in return_values]
	return_values_train = [i[1] for i in return_values]
	mae_train_values = [i[2] for i in return_values]
	mae_test_values = [i[3] for i in return_values]

	print(return_values)
	print("Average accuracy score for random forest test when depth is "+str(i)+": ",sum(return_values_test)/len(return_values_test))
	print("Average accuracy score for random forest train when depth is "+str(i)+": ",sum(return_values_train)/len(return_values_train))
	print("mean_absolute_error for random forest test when depth is "+str(i)+": ",sum(mae_test_values)/len(mae_test_values))
	print("mean_absolute_error for random forest train when depth is "+str(i)+": ",sum(mae_train_values)/len(mae_train_values))
	train_scores.append(sum(return_values_train)/len(return_values_train))
	test_scores.append(sum(return_values_test)/len(return_values_test))
	mae_train.append(sum(mae_train_values)/len(mae_train_values))
	mae_test.append(sum(mae_test_values)/len(mae_test_values))

values = [i for i in range(1, 21)]
pyplot.plot(values, mae_train, '-o', label='Train') # 
pyplot.plot(values, mae_test, '-o', label='Test')
pyplot.legend()
pyplot.title("Mean Absolute Error Random Forest")
pyplot.savefig("mae_random_forest_depth.png")
pyplot.show()

values = [i for i in range(1, 21)]
pyplot.plot(values, train_scores, '-o', label='Train') 
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.title("Accuracy Score Random Forest")
pyplot.savefig("as_random_forest_depth.png")
pyplot.show()

