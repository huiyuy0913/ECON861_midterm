# ECON861_midterm

## [part1_clean_data.py](https://github.com/huiyuy0913/ECON861_midterm/blob/main/part1_clean_data.py)

First, run part1_clean_data.py to clean the data.  I have also created a template named [clean_data_template.py](https://github.com/huiyuy0913/ECON861_midterm/blob/main/clean_data_template.py) based on this program.

## [part1_try_models.py](https://github.com/huiyuy0913/ECON861_midterm/blob/main/part1_try_models.py)
After running part1_try_models.py, which contains my rough selection process for models, I decided to choose the linear SVM model and the logistic model. I also created several graphs during this step.

## [part1_logistic_regression.py](https://github.com/huiyuy0913/ECON861_midterm/blob/main/part1_logistic_regression.py)

The next step is to execute part1_logistic_regression.py, where I created a pickle file that will be utilized in the subsequent programs.

## [part1_svm_linear.py](https://github.com/huiyuy0913/ECON861_midterm/blob/main/part1_svm_linear.py)

In this program, I identified the linear SVM model with the top 14 features from the logistic regression as the one with the highest accuracy score.

## [part2_make_prediction.py](https://github.com/huiyuy0913/ECON861_midterm/blob/main/part2_make_prediction.py)

In this program, I used the model trained above to make predictions on new customers' favorite games. However, I noticed that two observations in the new dataset contain empty cells. To fill them, I used the average numbers of the corresponding columns. As a result, the prediction for these two observations may not be accurate. These observations are located in row 612 and 2584.

I saved the original data along with the predicted game as a CSV file named [new_customers_dataset_with_predictions.csv](https://github.com/huiyuy0913/ECON861_midterm/blob/main/new_customers_dataset_with_predictions.csv).
