import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Paths
FEATURES_FILE = "../Extracted Features/tess_wav2vec_features.csv"
OUTPUT_MODEL_DIR = "../models/tess_simple_classifier/"
RESULTS_DIR = "../results/"
LOG_FILE = os.path.join(RESULTS_DIR, "training_log.txt")
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
EPOCHS = 20  # Define epochs here
LEARNING_RATE = 0.001  # Learning rate for the optimizer

# Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Simple Classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Load and Split Data
print("Loading features...")
df = pd.read_csv(FEATURES_FILE)
X = df.iloc[:, :-1].values
y = pd.factorize(df['label'])[0]  # Encode labels as integers

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
train_dataset = EmotionDataset(X_train, y_train)
val_dataset = EmotionDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleClassifier(input_dim=X.shape[1], num_classes=len(set(y))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Prepare Log File
log_file = open(LOG_FILE, "w")
log_file.write("Epoch,Train Loss,Val Loss,Accuracy,Precision,Recall,F1-Score\n")

# Training and Validation Loop
print("Training the model...")
train_losses, val_losses, accuracies, precisions, recalls, f1_scores = [], [], [], [], [], []

for epoch in range(EPOCHS):  # Use the centralized EPOCHS variable
    model.train()
    epoch_loss = 0
    y_train_true, y_train_pred = [], []

    # Training loop
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        y_train_true.extend(labels.cpu().numpy())
        y_train_pred.extend(preds.cpu().numpy())

    # Training Metrics
    train_accuracy = accuracy_score(y_train_true, y_train_pred)

    # Validation loop
    model.eval()
    val_loss = 0
    y_val_true, y_val_pred = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            val_loss += criterion(outputs, labels).item()

            preds = torch.argmax(outputs, dim=1)
            y_val_true.extend(labels.cpu().numpy())
            y_val_pred.extend(preds.cpu().numpy())

    # Validation Metrics
    val_accuracy = accuracy_score(y_val_true, y_val_pred)
    precision = precision_score(y_val_true, y_val_pred, average="weighted", zero_division=0)
    recall = recall_score(y_val_true, y_val_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_val_true, y_val_pred, average="weighted", zero_division=0)

    # Record Metrics
    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    accuracies.append(val_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    # Print and Log
    log_line = f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
    print(log_line)
    log_file.write(f"{epoch+1},{train_losses[-1]:.4f},{val_losses[-1]:.4f},{val_accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n")

log_file.close()
print(f"Training logs saved to: {LOG_FILE}")

# Save Model
torch.save(model.state_dict(), os.path.join(OUTPUT_MODEL_DIR, "simple_classifier.pth"))
print("Model saved successfully.")

# Save Metrics to CSV
metrics_df = pd.DataFrame({
    "Epoch": range(1, EPOCHS + 1),
    "Train Loss": train_losses,
    "Validation Loss": val_losses,
    "Accuracy": accuracies,
    "Precision": precisions,
    "Recall": recalls,
    "F1-Score": f1_scores
})
metrics_df.to_csv(os.path.join(RESULTS_DIR, "training_metrics.csv"), index=False)
print("Training metrics saved to CSV.")

# Plot Metrics
plt.figure(figsize=(12, 6))

# Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss", marker='x')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

# Accuracy, Precision, Recall, F1
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), accuracies, label="Accuracy", marker='o')
plt.plot(range(1, EPOCHS + 1), precisions, label="Precision", marker='x')
plt.plot(range(1, EPOCHS + 1), recalls, label="Recall", marker='^')
plt.plot(range(1, EPOCHS + 1), f1_scores, label="F1-Score", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Metrics")
plt.title("Validation Metrics")
plt.legend()
plt.grid()

# Save and Show
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_metrics_plot.png"))
plt.show()

# Final Evaluation on Validation Set
print("Evaluating final accuracy on the validation set...")
model.eval()
y_val_true, y_val_pred = [], []

with torch.no_grad():
    for features, labels in val_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        preds = torch.argmax(outputs, dim=1)
        y_val_true.extend(labels.cpu().numpy())
        y_val_pred.extend(preds.cpu().numpy())

# Compute Final Metrics
final_accuracy = accuracy_score(y_val_true, y_val_pred)
final_precision = precision_score(y_val_true, y_val_pred, average="weighted", zero_division=0)
final_recall = recall_score(y_val_true, y_val_pred, average="weighted", zero_division=0)
final_f1 = f1_score(y_val_true, y_val_pred, average="weighted", zero_division=0)

print(f"Final Accuracy: {final_accuracy * 100:.2f}%")
print(f"Final Precision: {final_precision:.4f}")
print(f"Final Recall: {final_recall:.4f}")
print(f"Final F1-Score: {final_f1:.4f}")

# Log Final Metrics
with open(LOG_FILE, "a") as log_file:
    log_file.write(f"\nFinal Metrics,Accuracy: {final_accuracy:.4f},Precision: {final_precision:.4f},"
                   f"Recall: {final_recall:.4f},F1-Score: {final_f1:.4f}\n")
