from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing, model_selection
from sklearn.metrics import precision_recall_fscore_support

# Load dataset (you can also load your own with pandas but sklearn offers
# a range of different datasets)
(X, y) = datasets.load_iris(return_X_y=True)

# No One-vs-All algorithm implemented yet, so just predicting two classe
y = y[0:100]
X = X[0:100, :]

# Preprocess it with sklearn (not necessary, but improves gradient descent)
X = preprocessing.scale(X)

# Divide dataset into train and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.65)

# Instantiate a new LogisticRegression and call its fit-method with the
# train data
lr = LogisticRegression()
lr.fit(X=X_train, y=y_train)

# Predict the result with the test data and calculate precision, recall
# and fscore
y_pred = lr.predict(X_test)
(precision, recall, fscore, _) = precision_recall_fscore_support(y_test, y_pred)

# Print interesting information
print(precision, recall, fscore)
print(lr.weights)

# Plot costs per iteration
plt.plot(lr.misclass_per_iter, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Total misclassifications')
plt.show()
