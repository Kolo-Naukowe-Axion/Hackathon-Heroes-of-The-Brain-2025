import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_PATH = '/home/pawel/neur/Neurohackathon/archive(1)/emotions.csv'
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Data Preprocessing ---
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"Data shape: {df.shape}")
    print(f"Classes: {y.unique()}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le

# --- Model Definition ---
class SimpleEmotionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super(SimpleEmotionMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.layer3 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        logits = self.layer3(x)
        return logits

# --- Training Loop ---
def train_model():
    # 1. Load Data
    X_train, X_test, y_train, y_test, le = load_and_preprocess_data(DATA_PATH)
    
    # 2. Create Datasets and Loaders
    train_dataset = EmotionDataset(X_train, y_train)
    test_dataset = EmotionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
    model = SimpleEmotionMLP(input_dim, num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    train_losses = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
    print("Training finished.")
    
    # --- Evaluation ---
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1) # Get probabilities for intensity
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))
    
    # Save model
    torch.save(model.state_dict(), 'emotion_model.pth')
    print("Model saved to emotion_model.pth")

if __name__ == "__main__":
    train_model()
