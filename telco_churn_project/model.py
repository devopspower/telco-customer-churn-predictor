import torch
import torch.nn as nn

class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): Number of input features from the dataset.
        """
        super(ChurnModel, self).__init__()
        
        # Analyze Logically: Architecture Design
        # We use a bottleneck-style architecture to extract high-level features
        self.network = nn.Sequential(
            # Layer 1: Expansion and Normalization
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64), # Stabilizes training by normalizing internal activations
            nn.ReLU(),
            nn.Dropout(0.3),    # Systematically prevents overfitting by 'turning off' 30% of neurons
            
            # Layer 2: Feature Refinement
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3: Output Projection
            nn.Linear(32, 1)    # Single output for binary classification (Logit)
        )
        
    def forward(self, x):
        """
        Performs a forward pass. 
        Note: We return 'logits' (raw values) rather than probabilities 
        to ensure numerical stability during training with BCEWithLogitsLoss.
        """
        return self.network(x)

# Logic for predicting probabilities (used during inference)
def get_predictions(model, input_tensor, threshold=0.5):
    """
    Converts model logits into binary predictions based on a business threshold.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).float()
    return predictions, probabilities