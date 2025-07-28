import glob
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
import os

def load_data(path):
    data = []
    labels = []
    for label in ['Cat', 'Dog']:
        for img_path in glob.glob(os.path.join(path, label, '*')):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (64,64))
            img = img / 255.0
            img = img.flatten()
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)

X_train, y_train = load_data('train')
X_test, y_test = load_data('test')

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print(f"KNN accuracy: {knn_acc:.3f}")

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
print(f"Logistic Regression accuracy: {log_acc:.3f}")

sgd = SGDClassifier(loss='log_loss', max_iter=1500)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)
sgd_acc = accuracy_score(y_test, sgd_pred)
print(f"SGD Classifier accuracy: {sgd_acc:.3f}")

results = {'KNN': knn_acc, 'LogisticRegression': log_acc, 'SGD': sgd_acc}
best_model_name = max(results, key=results.get)
if best_model_name == 'KNN':
    best_model = knn
elif best_model_name == 'LogisticRegression':
    best_model = log_reg
else:
    best_model = sgd

print(f"Best model: {best_model_name} ({results[best_model_name]:.3f})")
joblib.dump(best_model, 'catdog_model.z')

def predict_image(path, model):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (64,64))
    img_resized = img_resized / 255.0
    img_resized = img_resized.flatten().reshape(1, -1)
    pred = model.predict(img_resized)[0]
    cv2.putText(img, f'Pred: {pred}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_image(r'cat.jpg', best_model)