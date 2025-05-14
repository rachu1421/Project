# Projectfrom google.colab import files
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Upload the file from the local system
uploaded = files.upload()

# Then load the correct filename
df = pd.read_csv("road_safety_2024.csv.csv", low_memory=False)

# Drop unwanted columns
columns_to_drop = ['status', 'collision_index', 'collision_reference', 'vehicle_reference', 'lsoa_of_driver']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Replace -1 with NaN only in numeric columns
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].replace(-1, pd.NA)

# Drop rows where target ('escooter_flag') is missing
df.dropna(subset=['escooter_flag'], inplace=True)

# Fill missing values: median for numbers, mode for categorical
for col in df.columns:
    if df[col].dtype == 'object':
        # Fill missing values in categorical columns with mode
        mode_values = df[col].mode()
        if not mode_values.empty:  # Check if mode values exist
            df[col] = df[col].fillna(mode_values[0])  # Use the first mode value if available
    else:
        # Fill missing values in numeric columns with median
        df[col] = df[col].fillna(df[col].median())

# --- Data Visualization ---

# Bar plot: Top 10 vehicle types
if 'vehicle_type' in df.columns:
    plt.figure(figsize=(10, 5))
    df['vehicle_type'].value_counts().head(10).plot(kind='bar', color='pink')
    plt.title("Top 10 Vehicle Types Involved in Accidents")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Line plot: Accidents by year
if 'accident_year' in df.columns:
    plt.figure(figsize=(10, 5))
    df['accident_year'].value_counts().sort_index().plot(kind='line', marker='o', color='green')
    plt.title("Accidents Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Number of Accidents")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Supervised ML: Logistic Regression ---

# Split features and target
X = df.drop(columns=['escooter_flag'])
y = df['escooter_flag']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split with scaled data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model with class weight adjustment for imbalanced data
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
