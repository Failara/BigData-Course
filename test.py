import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = 1 - data.target

print("Data Loaded. Shape:", X.shape)
print("Class Balance:\n", pd.Series(y).value_counts().rename({0: 'Benign', 1: 'Malignant'}))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

metrics_data = []

def train_and_evaluate(model, X_tr, X_te, name):
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n--- {name} ---")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    metrics_data.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

train_and_evaluate(LogisticRegression(random_state=42), X_train_scaled, X_test_scaled, "Logistic Regression")
train_and_evaluate(SVC(kernel='linear', random_state=42), X_train_scaled, X_test_scaled, "SVM (Linear)")
train_and_evaluate(SVC(kernel='rbf', random_state=42), X_train_scaled, X_test_scaled, "SVM (RBF)")
train_and_evaluate(RandomForestClassifier(random_state=42), X_train, X_test, "Random Forest")

results_df = pd.DataFrame(metrics_data).set_index('Model')
print("\n=== Final Metrics Comparison ===")
print(results_df)