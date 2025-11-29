import joblib
import sys

try:
    le = joblib.load('/home/pawel/neur/Neurotes1/Neurohackathon/backends/Neurohackathon-pawel/label_encoder.pkl')
    print(f"Classes: {le.classes_}")
except Exception as e:
    print(f"Error: {e}")
