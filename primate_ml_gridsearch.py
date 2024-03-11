import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

# Load data
with open("primate_dataset.json", "r") as file:
    data = json.load(file)
texts = [item['post_text'] for item in data]

# Convert annotations to list of symptoms with "yes" label
labels = [[symptom[0] for symptom in post['annotations'] if symptom[1] == 'yes'] for post in data]

# Convert labels to binary
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
symptom_names = mlb.classes_

# Convert texts to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Define the models
log_reg = OneVsRestClassifier(LogisticRegression())
rand_forest = OneVsRestClassifier(RandomForestClassifier())
svm = OneVsRestClassifier(SVC())

# # Train and evaluate each model
# for model in [log_reg, rand_forest, svm]:
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred, target_names=symptom_names))

# Define the models
models = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression()),
    "Random Forest": OneVsRestClassifier(RandomForestClassifier()),
    "SVM": OneVsRestClassifier(SVC()),
    "XGBoost": OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))  # Added XGBoost
}

# Define the hyperparameters
hyperparameters = {
    "Logistic Regression": {
        'estimator__C': [0.1, 1, 10],
        'estimator__penalty': ['l1', 'l2']
    },
    "XGBoost": {
        'estimator__n_estimators': [100, 200],
        'estimator__learning_rate': [0.01, 0.1],
        'estimator__max_depth': [3, 5, 7, 10],
        'estimator__min_child_weight': [1, 3, 5]
    },
    "Random Forest": {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [None, 5, 10],
        'estimator__min_samples_split': [2, 5, 10]
    },
    "SVM": {
        'estimator__C': [0.1, 1, 10],
        'estimator__kernel': ['linear', 'rbf']
    }
}


# Train, tune and evaluate each model
for model_name, model in models.items():
    print('\n=====================================================')
    print(f"\nTraining and tuning {model_name}...")
    
    # Create the GridSearchCV object
    grid = GridSearchCV(model, hyperparameters[model_name], cv=5, scoring='f1_micro')
    
    # Fit the data to the GridSearchCV object and find the best parameters
    grid.fit(X_train, y_train)
    
    # Print the best parameters
    print(f"Best parameters for {model_name}: {grid.best_params_}")
    
    # Evaluate the model with the best parameters on the test set
    y_pred = grid.predict(X_test)
    print(f"\nClassification report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=symptom_names))
    print('=====================================================')

    # Calculate and print MCC for each symptom
    for i in range(y_test.shape[1]):
        mcc = matthews_corrcoef(y_test[:, i], y_pred[:, i])
        print(f"MCC for {symptom_names[i]}: {mcc}")