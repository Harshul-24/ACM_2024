import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import matthews_corrcoef
from gensim.models import KeyedVectors
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

# Load BioWord2Vec
model = KeyedVectors.load_word2vec_format('BioWordVec_PubMed_MIMICIII_d200.vec.bin', binary=True)

# Define a function to create a feature vector for each text
def get_feature_vector(text):
    words = text.split()
    feature_vector = np.zeros((200,), dtype="float32")  # BioWord2Vec vectors have 200 dimensions
    num_words = 0
    for word in words:
        if word in model:
            num_words += 1
            feature_vector = np.add(feature_vector, model[word])
    if num_words > 0:
        feature_vector = np.divide(feature_vector, num_words)
    return feature_vector

# Create a feature vector for each text
X = np.array([get_feature_vector(text) for text in texts])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Define the models
models = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression()),
    "Random Forest": OneVsRestClassifier(RandomForestClassifier()),
    "SVM": OneVsRestClassifier(SVC()),
    "XGBoost": OneVsRestClassifier(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))  # Added XGBoost
}

# Train and evaluate each model
for model_name, model in models.items():
    print('\n=====================================================')
    print(f"\nTraining and evaluating {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nClassification report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=symptom_names))
    print('=====================================================')

    # Calculate and print MCC for each symptom
    for i in range(y_test.shape[1]):
        mcc = matthews_corrcoef(y_test[:, i], y_pred[:, i])
        print(f"MCC for {symptom_names[i]}: {mcc}")
    
    print('=====================================================')



