import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_and_save_model(data_filename="noise_data.csv"):
    """
    Loads the dataset, trains the model and scaler, and saves them
    as .pkl files.
    """
    # --- 1. Data Loading and Preprocessing ---
    if not os.path.exists(data_filename):
        print(f"--- ERROR: The data file '{data_filename}' was not found. ---")
        print("Please run the `prepare_data.py` script first to create it.")
        exit()

    print("Loading the prepared dataset...")
    df = pd.read_csv(data_filename)

    print("Encoding categorical features...")
    # Use a dictionary to store encoders for deployment consistency
    encoders = {}
    categorical_cols = ['City', 'State', 'Type']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le # Save the encoder

    # --- 2. Model Training ---
    print("Splitting data into training and testing sets...")
    X = df[['Year', 'Month', 'Day', 'Night', 'City', 'State', 'Type']]
    y = df['NoiseLevel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Training the logistic regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # --- 3. Save the Artifacts ---
    print("Saving the trained scaler to 'scaler.pkl'...")
    joblib.dump(scaler, 'scaler.pkl')

    print("Saving the trained model to 'logistic_model.pkl'...")
    joblib.dump(model, 'logistic_model.pkl')

    print("\n--- Training complete. 'scaler.pkl' and 'logistic_model.pkl' have been created. ---")


if __name__ == '__main__':
    # This function will run when you execute `python train.py`
    train_and_save_model()

