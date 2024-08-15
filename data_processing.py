'''
This file is used to manipulate the data in the csv file and to 
make the model for predictions. Both are stored in local files for quick access later.
'''

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from joblib import dump, load
import os

def categorize(val):
	# turn each value into a category 0-4
	bins=5
	bin_size = 63771/bins 
	return val // bin_size

if not os.path.exists('data.csv'):
	df = pd.read_csv('insurance.csv')
	df['sex'] = df['sex'].replace({'male':1, 'female':0})
	df['smoker'] = df['smoker'].replace({'yes':1, 'no':0})
	df['region'] = df['region'].replace({
			'southwest': 0,
			'southeast': 1,
			'northwest': 2,
			'northeast': 3
		})
	df['charges'] = df['charges'].apply(categorize).astype('category')

	df.to_csv('data.csv', index=False)
else:
	df = pd.read_csv('data.csv')
	
X_train, X_test, y_train, y_test = train_test_split(df.drop(['charges'], axis=1), df['charges'], test_size=0.2)
print(X_train.sample(10))

import warnings
warnings.filterwarnings('ignore')

if os.path.exists('model.joblib'):
	model = load('model.joblib')
	print('loaded')
else:
	model = MLPClassifier(hidden_layer_sizes=(32, 32), 
			activation='relu', 
			solver='adam', 
			max_iter=5000,
			learning_rate='constant'
			)
	model.fit(X_train, y_train)

	dump(model, 'model.joblib')

# show 4 previous datapoints
fig, ax = plt.subplots(2, 2)

for i in range(2):
	for j in range(2):
		print(y_test.iloc[i+j])
		ax[i][j].bar(
					['<12754.2', '<25508.4', '<38262.6', '<51016.8', '<63771.0'], 
					[y_test.iloc[i+j] == x for x in range(5)]
				)
		ax[i][j].set_xticklabels(['<12754.2', '<25508.4', '<38262.6', '<51016.8', '<63771.0'], rotation=45)


# predict a new data point
plt.bar(range(5), model.predict_proba(X_test)[0])
plt.show()