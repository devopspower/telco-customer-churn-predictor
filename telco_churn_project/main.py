import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import our custom modules
from dataset import TelcoDataset
from model import ChurnModel
from train import train_one_epoch, validate

def run_pipeline(data_path, epochs=30, batch_size=64, lr=0.001):
    """
    Orchestrates the full machine learning workflow.
    """
    # 1. Define Device (Use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Ground Objectively: Load Datasets
    # Ensure you have downloaded the CSV from Kaggle and placed it in the path below
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print("Loading and preprocessing data...")
    train_set = TelcoDataset(data_path, mode='train')
    test_set = TelcoDataset(data_path, mode='test')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 3. Analyze Logically: Initialize Model, Loss, and Optimizer
    input_dim = train_set.get_input_dims()
    model = ChurnModel(input_dim).to(device)
    
    # We use BCEWithLogitsLoss because our model outputs raw logits
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Explore Systematically: The Training Loop
    print(f"Starting training for {epochs} epochs...")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        # Log progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        # Save the best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_churn_model.pth")

    # 5. Validate Rigorously: Final Report
    print("-" * 30)
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2%}")
    print("Model weights saved as 'best_churn_model.pth'")

if __name__ == "__main__":
    # Path to the Kaggle dataset
    DATA_FILE = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    run_pipeline(DATA_FILE)