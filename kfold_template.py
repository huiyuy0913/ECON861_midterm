# from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import mean_absolute_error

def run_kfold(data, target, machine, n, use_r2=True, use_accuracy=False, use_confusion=False):

	# print("run kfold")
	kfold_object = KFold(n_splits=n)
	kfold_object.get_n_splits(data)

	# print(kfold_object)


	
	all_return_values = []
	i = 0
	for train_index, test_index in kfold_object.split(data):
		i=i+1
		# print("Round:", str(i))
		# print("Training index:")
		# print(train_index)
		# print("Testing index")
		# print(test_index)
		
		data_train = data[train_index]
		target_train = target[train_index]
		data_test = data[test_index]
		target_test = target[test_index]
		# machine = linear_model.LinearRegression()
		machine.fit(data_train, target_train)

		prediction = machine.predict(data_test)
		target_yhat = machine.predict(data_train)
		return_value = []
		if (use_r2 == True):	
			r2 = metrics.r2_score(target_test, prediction)
			# print("R square score: ",r2)
			return_value.append(r2)
		if (use_accuracy == True):
			accuracy = metrics.accuracy_score(target_test, prediction)
			train_acc = metrics.accuracy_score(target_train, target_yhat)
			# print("Accuracy score: ",accuracy)
			return_value.append(accuracy)
			return_value.append(train_acc)
			return_value.append(mean_absolute_error(target_train, target_yhat))
			return_value.append(mean_absolute_error(target_test, prediction))
		# print(prediction)
		# print(target_test)
		if (use_confusion == True):
			confusion = metrics.confusion_matrix(target_test, prediction)
			# print("Confusion matrix:\n ", confusion)
			return_value.append(confusion)

		# print("\n\n") # means two new lines
		all_return_values.append(return_value)
	return all_return_values


if __name__ == '__main__':
    run_kfold()
