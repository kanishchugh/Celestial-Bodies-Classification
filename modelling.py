import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf

# Read the data
data_path = "./data/"
data = pd.read_csv(data_path + "data_preprocessed.csv")

# Split the data into training and test sets
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logistic_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train_scaled.shape[1],))
])

logistic_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
logistic_model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, verbose=0)

# Evaluate Logistic Regression
prediction_logistic = (logistic_model.predict(X_test_scaled) > 0.5).astype(int).reshape(-1)

precision_logistic = precision_score(y_test, prediction_logistic)
recall_logistic = recall_score(y_test, prediction_logistic)
F1measure_logistic = f1_score(y_test, prediction_logistic)
accuracy_logistic = accuracy_score(y_test, prediction_logistic)
auc_logistic = roc_auc_score(y_test, logistic_model.predict(X_test_scaled).ravel())

print('Logistic Regression:')
print('Precision:', precision_logistic)
print('Recall:', recall_logistic)
print('F1 Score:', F1measure_logistic)
print('Accuracy:', accuracy_logistic)
print('AUC:', auc_logistic)

# Decision Tree
decision_tree_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

decision_tree_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
decision_tree_model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, verbose=0)

# Evaluate Decision Tree
prediction_dt = (decision_tree_model.predict(X_test_scaled) > 0.5).astype(int).reshape(-1)

precision_dt = precision_score(y_test, prediction_dt)
recall_dt = recall_score(y_test, prediction_dt)
F1measure_dt = f1_score(y_test, prediction_dt)
accuracy_dt = accuracy_score(y_test, prediction_dt)

print('Decision Tree:')
print('Precision:', precision_dt)
print('Recall:', recall_dt)
print('F1 Score:', F1measure_dt)
print('Accuracy:', accuracy_dt)

# Random Forest
random_forest_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train_scaled.shape[1],))
])

random_forest_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
random_forest_model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, verbose=0)

# Evaluate Random Forest
prediction_rf = (random_forest_model.predict(X_test_scaled) > 0.5).astype(int).reshape(-1)

precision_rf = precision_score(y_test, prediction_rf)
recall_rf = recall_score(y_test, prediction_rf)
F1measure_rf = f1_score(y_test, prediction_rf)
accuracy_rf = accuracy_score(y_test, prediction_rf)

print('Random Forest:')
print('Precision:', precision_rf)
print('Recall:', recall_rf)
print('F1 Score:', F1measure_rf)
print('Accuracy:', accuracy_rf)

# Neural Network
neural_network_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train_scaled.shape[1],))
])

neural_network_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
neural_network_model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, verbose=0)

# Evaluate Neural Network
prediction_nn = (neural_network_model.predict(X_test_scaled) > 0.5).astype(int).reshape(-1)

precision_nn = precision_score(y_test, prediction_nn)
recall_nn = recall_score(y_test, prediction_nn)
F1measure_nn = f1_score(y_test, prediction_nn)
accuracy_nn = accuracy_score(y_test, prediction_nn)

print('Neural Network:')
print('Precision:', precision_nn)
print('Recall:', recall_nn)
print('F1 Score:', F1measure_nn)
print('Accuracy:', accuracy_nn)
