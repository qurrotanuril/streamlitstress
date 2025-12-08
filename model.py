import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pickle

# LOAD DATASET
data = pd.read_csv("Student Stress Factors (2).csv")

data = data.rename(columns={
    "Kindly Rate your Sleep Quality 😴": "sleep_quality",
    "How many times a week do you suffer headaches 🤕?": "headache_freq",
    "How would you rate you academic performance 👩‍🎓?": "academic_perf",
    "how would you rate your study load?": "study_load",
    "How many times a week you practice extracurricular activities 🎾?": "extracurricular_freq",
    "How would you rate your stress levels?": "stress_level"
})

data = data.drop(columns=["Timestamp"])

# SPLIT
X = data.drop(columns=["stress_level"])
y = data["stress_level"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# SAVE
pickle.dump(model, open("stress_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model baru berhasil dilatih & disimpan!")
