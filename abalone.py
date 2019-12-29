import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC


data = pd.read_csv("/home/ubuntu/PycharmProjects/supervised_learning/abalone/abalone.data")

#create traina and test sets
y = data["Sex"]
X = data.drop("Sex", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 50)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)


    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)


# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1,50)}
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)

print(knn_cv.best_params_)
print(knn_cv.best_score_)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100000)

s= pd.DataFrame(data["Sex"])
s['Sex'] = s.apply(lambda x: 0 if x['Sex'] == "M" else (1 if x['Sex'] == "F" else 2), axis=1)
y=s["Sex"]

cv_results = cross_val_score(logreg, X,y, cv=5)
print(cv_results)
print(np.mean(cv_results))

svm = LinearSVC(max_iter=100000)
cv_results_2 = cross_val_score(svm, X,y, cv=5)
print(cv_results_2)
print(np.mean(cv_results_2))
