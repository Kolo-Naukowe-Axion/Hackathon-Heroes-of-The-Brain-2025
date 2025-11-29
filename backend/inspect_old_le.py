import joblib
import sys

try:
    le = joblib.load('/Users/iwosmura/axion antigra/backend/label_encoder.pkl')
    print("Old Label Encoder Classes:", le.classes_)
except Exception as e:
    print(f"Error loading old label encoder: {e}")
