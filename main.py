import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor ,RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score ,mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt
data=pd.read_csv('Pet_Cons_Dec_Tree.csv')
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=1)
dtree=DecisionTreeRegressor(max_depth=3,min_samples_split=5,min_samples_leaf=5,max_leaf_nodes=5,random_state=0)
dtree.fit(x_train,y_train)
y_pre=dtree.predict(x_test)
x_train_fit=dtree.predict(x_train)
tree.plot_tree(dtree)
plt.show()
feature_names=['Petrol_tax','Average_income','Paved_Highways','Population_Driver']
target=['Petrol_Consumption']
tree.plot_tree(dtree,feature_names=feature_names,class_names=target,filled=True,rounded=True)
plt.show()
#print("mean_squared_error(train):",np.sqrt(mean_squared_error(y_train,x_train_fit)))
print("mean_squared_error(test):",np.sqrt(mean_squared_error(y_test,y_pre)))
print("r2_score(test)",r2_score(y_test,y_pre))
#print("r2_score(train)",r2_score(y_train,x_train_fit))




