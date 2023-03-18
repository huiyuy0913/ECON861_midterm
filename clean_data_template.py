import pandas
import numpy

def clean_data(dataset):

	dataset = dataset.dropna().reset_index(drop=True)
	dataset = dataset[dataset['age']>=0]
	numeric_col = pandas.to_numeric(dataset['personality2'], errors='coerce')
	non_numeric_rows = dataset[numeric_col.isna()]
	numeric_col = pandas.to_numeric(dataset['personality4'], errors='coerce')
	non_numeric_rows = dataset[numeric_col.isna()]
	numeric_col = pandas.to_numeric(dataset['personality5'], errors='coerce')
	non_numeric_rows = dataset[numeric_col.isna()]
	replace_dict_2 = {'I': 1}
	replace_dict_45 = {'O': 0}
	dataset['personality2'] = dataset['personality2'].replace(replace_dict_2, regex=True).astype(float)
	dataset['personality4'] = dataset['personality4'].replace(replace_dict_45, regex=True).astype(float)
	dataset['personality5'] = dataset['personality5'].replace(replace_dict_45, regex=True).astype(float)
	return dataset

if __name__ == '__main__':
    clean_data()
