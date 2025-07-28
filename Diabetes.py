#Diabetes Risk Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Load and preview data
df = pd.read_csv(r"D:\Diabetes\diabetes.csv")  

print("First 5 rows of data:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())  

#Prepare features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

#Split into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

#Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Build Logistic Regression model
log_reg = LogisticRegression(max_iter=200, random_state=42)
log_reg.fit(X_train_scaled, y_train)

#Build Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

#Predict on test data
y_pred_logreg = log_reg.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test_scaled)

#Evaluate models
def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"\n--- {model_name} Evaluation ---")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

evaluate_model(y_test, y_pred_logreg, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

#Feature Importance - Logistic Regression coefficients
logreg_coef = pd.Series(log_reg.coef_[0], index=X.columns).sort_values(ascending=False)
print("\nLogistic Regression Feature Importance (Coefficients):")
print(logreg_coef)

plt.figure(figsize=(10,6))
logreg_coef.plot(kind='bar')
plt.title('Logistic Regression Feature Importance')
plt.ylabel('Coefficient Value')
plt.xlabel('Feature')
plt.show()

#Feature Importance - Random Forest
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom Forest Feature Importance:")
print(rf_importance)

plt.figure(figsize=(10,6))
rf_importance.plot(kind='bar')
plt.title('Random Forest Feature Importance')
plt.ylabel('Feature Importance')
plt.xlabel('Feature')
plt.show()

