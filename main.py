import pandas as pd
df = pd.read_csv("datasets_228_482_diabetes.csv")
A = df.isnull().sum()
print(df)
print(A)

import matplotlib.pyplot  as plt
plt.hist(['pregnant','BP','Insulin','Diabetes','Outcome','Glucose','SkinT','BMI','Age'],bins=17,color=['blue'])
plt.show()

plt.scatter(x=df['pregnant'],y=df['Outcome'],c='blue')
plt.scatter(x=df['BMI'],y=df['Insulin'],c='red')
plt.scatter(x=df['Glucose'],y=df['Diabetes'],c='yellow')
plt.scatter(x=df['BP'],y=df['Age'],c='pink')
plt.scatter(x=df['Diabetes'],y=df['Outcome'],c='black')

from sklearn.model_selection import train_test_split
feature_cols=['pregnant','Glucose','BP','SkinT','Insulin','BMI','Diabetes','Age']
x = df[feature_cols]
y = df.Outcome
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)
print(x_train,x_test,y_train,y_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
y_predict = knn.predict(x_test)
print(x_train,x_test,y_train,y_test)

from sklearn import metrics
print("Accuracy is:",metrics.accuracy_score(y_test,y_predict))

from sklearn.tree import DecisionTreeClassifier
RA = DecisionTreeClassifier()
RA.fit(x_train,y_train)
y_predict = RA.predict(x_test)
print("Accuracy is:",metrics.accuracy_score(y_test,y_predict))
print(x_train,x_test,y_train,y_test)

print(knn.score(x_test,y_test))

data = pd.read_csv("newdataset.csv")
print(knn.predict(data))


































