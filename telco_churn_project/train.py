import torch
import torch.nn as nn

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles the training logic for a single pass through the dataset.
    """
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(loader):
        # Move data to GPU or CPU
        data, target = data.to(device), target.to(device)
        
        # 1. Clear gradients
        optimizer.zero_grad()
        
        # 2. Forward pass: Compute predicted outputs by passing inputs to the model
        outputs = model(data)
        
        # 3. Calculate Loss
        loss = criterion(outputs, target)
        
        # 4. Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # 5. Perform a single optimization step (parameter update)
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    """
    Evaluates the model on unseen data to ensure generalization.
    """
    model.eval()
    val_loss = 0
    correct = 0
    
    # Disable gradient calculation for efficiency and to prevent leakage
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            val_loss += criterion(outputs, target).item()
            
            # Convert logits to probabilities and then to binary (0 or 1)
            # Probability > 0.5 is predicted as 'Churn'
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Count correct predictions
            correct += (preds == target).sum().item()
            
    avg_loss = val_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    
    return avg_loss, accuracy