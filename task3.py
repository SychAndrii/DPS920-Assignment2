import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


# Upload them locally, since GitHub does not allow large files to be stored in the cloud.
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

X_train = train_data.iloc[:, 1:].values / 255.0
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values / 255.0
y_test = test_data.iloc[:, 0].values

print("Train:", X_train.shape, "Test:", X_test.shape)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))
print(f"KNN accuracy: {knn_acc:.3f}")

sgd = SGDClassifier(loss='log_loss', max_iter=1000)
sgd.fit(X_train, y_train)
sgd_acc = accuracy_score(y_test, sgd.predict(X_test))
print(f"SGD accuracy: {sgd_acc:.3f}")

results = {'KNN': knn_acc, 'SGD': sgd_acc}
best_model_name = max(results, key=results.get)

if best_model_name == 'KNN':
    best_model = knn
else:
    best_model = sgd

print(f"Best model: {best_model_name} ({results[best_model_name]:.3f})")
joblib.dump(best_model, 'mnist_model.z')

def predict_sample(index):
    img = X_test[index].reshape(28,28)
    pred = best_model.predict([X_test[index]])[0]
    print(f"True label: {y_test[index]}, Predicted: {pred}")

predict_sample(5)
predict_sample(8)
predict_sample(22)