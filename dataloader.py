import pandas as pd
import numpy as np
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import json
import warnings

# Suppress all future warnings for cleaner output
warnings.filterwarnings("ignore")

# --- DATA CONSOLIDATION FROM JSON ---
# We will manually load and parse the JSON files provided.
json_data_content = {
    "all_data_20250817_095650.json": """
    {
      "outputs": [
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.376, "quantum_mate": 0.328 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 2, "quantum_host": 0.264, "quantum_mate": 0.320 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 3, "quantum_host": 0.424, "quantum_mate": 0.416 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 4, "quantum_host": 0.208, "quantum_mate": 0.256 }}}
      ]
    }
    """,
    "all_data_20250817_095811.json": """
    {
      "outputs": [
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.288, "quantum_mate": 0.312 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 2, "quantum_host": 0.208, "quantum_mate": 0.240 }}}
      ]
    }
    """,
    "all_data_20250817_095959.json": """
    {
      "outputs": []
    }
    """,
    "all_data_20250817_100018.json": """
    {
      "outputs": [
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.240, "quantum_mate": 0.264 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.336, "quantum_mate": 0.320 }}},
        { "parsed_data": { "quantum_sentiment": { "cycle": 1, "quantum_host": 0.208, "quantum_mate": 0.256 }}}
      ]
    }
    """
}

data_list = []
for file_name, content in json_data_content.items():
    data = json.loads(content)
    for output in data['outputs']:
        if output.get('parsed_data') and 'quantum_sentiment' in output['parsed_data']:
            cycle_data = output['parsed_data']['quantum_sentiment']
            new_row = {
                'Cycle': cycle_data['cycle'],
                'Collective Coherence': cycle_data['quantum_host'],
                'Hive Synchrony': cycle_data['quantum_mate']
            }
            data_list.append(new_row)

df = pd.DataFrame(data_list)
# Clean and sort the consolidated data
df = df.drop_duplicates(subset=['Cycle', 'Collective Coherence', 'Hive Synchrony']).sort_values(by='Cycle').reset_index(
    drop=True)

# Add derivatives to the new dataframe
df['Coherence Derivative'] = df['Collective Coherence'].diff().fillna(0)
df['Synchrony Derivative'] = df['Hive Synchrony'].diff().fillna(0)
# Create a binary label for coherence change: 1 for increase, 0 for decrease/no change
df['Coherence Change'] = (df['Coherence Derivative'] > 0).astype(int)

print("--- CONSOLIDATED DATA ---")
print(df.head())
print("\n--- DERIVATIVES and LABELS ---")
print(df[['Coherence Derivative', 'Coherence Change']].head())

# --- EMODEL (PERCEPTRON) DEFINITION AND TRAINING ---
# Define features (X) and labels (y) for the model
X = df[['Collective Coherence', 'Hive Synchrony']].values
y = df['Coherence Change'].values

# Normalize the input features
scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.from_numpy(X_normalized).float()
y_tensor = torch.from_numpy(y).long()


# Our Emodel is a simple neural network for classification
class Emodel(nn.Module):
    def __init__(self, input_dim):
        super(Emodel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 2)  # Two outputs for binary classification

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


# Hyperparameters and model setup
input_dim = X_tensor.shape[1]
epochs = 500
emodel = Emodel(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(emodel.parameters(), lr=0.01)

# Training loop for the Emodel
print("\n--- TRAINING EMODEL ---")
for epoch in range(epochs):
    outputs = emodel(X_tensor)
    loss = criterion(outputs, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print("\nTraining finished! Emodel is ready to make predictions.")


# --- EMODEL DEMONSTRATION ---
def demonstrate_emodel_prediction(model, scaler):
    """
    Simulates a new data point and asks the emodel to predict the outcome.
    """
    print("\n--- EMODEL PREDICTION ---")
    # Take the latest data point from our new consolidated data
    last_state = df.iloc[-1][['Collective Coherence', 'Hive Synchrony']].values.reshape(1, -1)

    print(f"Current state: Coherence={last_state[0][0]:.3f}, Synchrony={last_state[0][1]:.3f}")

    # Normalize the data point and convert to tensor
    last_state_normalized = scaler.transform(last_state)
    last_state_tensor = torch.from_numpy(last_state_normalized).float()

    # Get the prediction from the Emodel
    with torch.no_grad():
        model.eval()
        output = model(last_state_tensor)
        _, predicted_label = torch.max(output.data, 1)

    # Interpret the prediction
    if predicted_label.item() == 1:
        print("Emodel prediction: The collective coherence is likely to INCREASE. 📈")
    else:
        print("Emodel prediction: The collective coherence is likely to DECREASE. 📉")


# Run the demonstration
demonstrate_emodel_prediction(emodel, scaler_X)
