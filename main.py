# performing linear algebra 
import numpy as np 
# data processing 
import pandas as pd 
# visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 

dataset = pd.read_csv("DATASET/WA_Fn-UseC_-HR-Employee-Attrition.csv") 
print (dataset.head)
dataset.info()

# heatmap to check the missing value 
plt.figure(figsize =(10, 4)) 
sns.heatmap(dataset.isnull(), 
			yticklabels = False, 
			cbar = False, 
			cmap ='viridis')

sns.set_style('darkgrid') 
sns.countplot(x ='Attrition', 
			data = dataset) 

sns.lmplot(x = 'Age', 
		y = 'DailyRate', 
		hue = 'Attrition', 
		data = dataset) 
plt.figure(figsize =(10, 6)) 
sns.boxplot(y ='MonthlyIncome', 
			x ='Attrition', 
			data = dataset) 


dataset.drop('EmployeeCount', 
			axis = 1, 
			inplace = True) 
dataset.drop('StandardHours', 
			axis = 1, 
			inplace = True) 
dataset.drop('EmployeeNumber', 
			axis = 1, 
			inplace = True) 
dataset.drop('Over18', 
			axis = 1, 
			inplace = True) 

print(dataset.shape)

# encoding the categorical data
dataset = pd.get_dummies(dataset, 
                        drop_first = True) 
print(dataset.shape)
dataset.head()

X = dataset.drop('Attrition_Yes', 
                axis = 1)
y = dataset['Attrition_Yes']

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( 

X, y, test_size = 0.25, random_state = 40)

from sklearn.model_selection import cross_val_predict, cross_val_score 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier(n_estimators=10, 
							criterion='entropy') 
rf.fit(X_train, y_train) 
y_pred = rf.predict(X_test) 

def print_score(clf, X_train, y_train, 
				X_test, y_test, 
				train=True): 

	if train: 
		print("Train Result:") 
		print("------------") 
		print("Classification Report: \n {}\n".format(classification_report( 
			y_train, clf.predict(X_train)))) 

		print("Confusion Matrix: \n {}\n".format(confusion_matrix( 
			y_train, clf.predict(X_train)))) 

		res = cross_val_score(clf, X_train, y_train, 
							cv=10, scoring='accuracy') 

		print("Average Accuracy: \t {0:.4f}".format(np.mean(res))) 
		print("Accuracy SD: \t\t {0:.4f}".format(np.std(res))) 
		print("----------------------------------------------------------") 

	elif train == False: 

		print("Test Result:") 
		print("-----------") 
		print("Classification Report: \n {}\n".format( 
			classification_report(y_test, clf.predict(X_test)))) 

		print("Confusion Matrix: \n {}\n".format( 
			confusion_matrix(y_test, clf.predict(X_test)))) 

		print("accuracy score: {0:.4f}\n".format( 
			accuracy_score(y_test, clf.predict(X_test)))) 

		print("-----------------------------------------------------------") 

print_score(rf, X_train, y_train, 
			X_test, y_test, 
			train=True) 

print_score(rf, X_train, y_train, 
			X_test, y_test, 
			train=False) 

