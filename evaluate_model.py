import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    # Load dataset
    df = pd.read_csv("dataset-pp.csv")

    # Load encoders and scaler
    gender_encoder = joblib.load("gender_encoder.pkl")
    occupation_encoder = joblib.load("occupation_encoder.pkl")
    location_encoder = joblib.load("location_encoder.pkl")
    stress_encoder = joblib.load("stress_encoder.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")

    # Encode categorical columns
    df['Gender'] = gender_encoder.transform(df['Gender'])
    df['Occupation'] = occupation_encoder.transform(df['Occupation'])
    df['Location'] = location_encoder.transform(df['Location'])
    df['StressLevel'] = stress_encoder.transform(df['StressLevel'])
    df['MentalHealthRisk'] = target_encoder.transform(df['MentalHealthRisk'])

    # Prepare features and target
    # Updated column names to match the dataset
    X = df[['Age', 'Gender', 'Occupation', 'Location', 'SleepHours', 'ScreenTime', 'PhysicalActivity', 'PHQ9_Score', 'GAD7_Score', 'StressLevel']]
    y = df["MentalHealthRisk"]

    # Scale the data
    X_scaled = scaler.transform(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    print("🔍 Model Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Regression-style metrics
    print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

# Run only if this file is executed directly
if __name__ == "__main__":
    evaluate_model()
