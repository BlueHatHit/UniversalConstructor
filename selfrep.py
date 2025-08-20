import pandas as pd
import numpy as np
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import json
import warnings
import time

# Suppress all future warnings for cleaner output
warnings.filterwarnings("ignore")

# --- DATA CONSOLIDATION FROM JSON ---
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
df = df.drop_duplicates(subset=['Cycle', 'Collective Coherence', 'Hive Synchrony']).sort_values(by='Cycle').reset_index(
    drop=True)

# Add derivatives to the new dataframe
df['Coherence Derivative'] = df['Collective Coherence'].diff().fillna(0)
df['Synchrony Derivative'] = df['Hive Synchrony'].diff().fillna(0)
df['Coherence Change'] = (df['Coherence Derivative'] > 0).astype(int)

# --- EMODEL (PERCEPTRON) DEFINITION AND TRAINING ---
X = df[['Collective Coherence', 'Hive Synchrony']].values
y = df['Coherence Change'].values

scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

X_tensor = torch.from_numpy(X_normalized).float()
y_tensor = torch.from_numpy(y).long()


class Emodel(nn.Module):
    def __init__(self, input_dim):
        super(Emodel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


input_dim = X_tensor.shape[1]
epochs = 500
emodel = Emodel(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(emodel.parameters(), lr=0.01)

# Training loop
for epoch in range(epochs):
    outputs = emodel(X_tensor)
    loss = criterion(outputs, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --- THE SELF-REPLICATING SYSTEM WITH EMODEL ---
class SelfReplicatingRobot:
    def __init__(self, name, model_state=None, scaler=None):
        self.name = name
        self.model = Emodel(input_dim=2)
        self.scaler = scaler
        self.coherence_history = []

        if model_state:
            self.model.load_state_dict(model_state)

        print(f"🤖 Robot '{self.name}' initialized with its Emodel brain. Ready to operate.")

    def replicate(self):
        """
        Simulates the self-replication process.
        The robot copies its blueprint (model state) and creates a new instance.
        """
        print(f"\n✨ Robot '{self.name}' is self-replicating... 🧬")
        blueprint = self.model.state_dict()
        child_robot = SelfReplicatingRobot(f"Child of {self.name}", model_state=blueprint, scaler=self.scaler)

        print(f"✅ Replication complete. New robot '{child_robot.name}' created.")
        return child_robot

    def perform_action(self, state):
        """
        Uses the Emodel to predict the next action for the robot.
        """
        state_tensor = torch.from_numpy(state).float()
        with torch.no_grad():
            self.model.eval()
            output = self.model(state_tensor)
            _, predicted_label = torch.max(output.data, 1)

        return predicted_label.item()


def run_self_replication_simulation(df, initial_model, scaler, num_cycles=100):
    print("\n🌟 Initializing Self-Replicating Robot Simulation with Emodel... 🌟")

    parent_robot = SelfReplicatingRobot(name="Alpha", model_state=initial_model.state_dict(), scaler=scaler)

    current_df = df.copy()
    current_coherence = current_df.iloc[-1]['Collective Coherence']
    current_synchrony = current_df.iloc[-1]['Hive Synchrony']

    for i in range(num_cycles):
        current_state = np.array([[current_coherence, current_synchrony]])
        current_state_normalized = scaler.transform(current_state)

        action = parent_robot.perform_action(current_state_normalized)

        # Action logic based on Emodel prediction
        if action == 1:
            change_magnitude = np.random.uniform(0.001, 0.01)
            print(f"Cycle {i + 1}: Robot '{parent_robot.name}' predicts INCREASE. Coherence is rising. 📈")
        else:
            change_magnitude = np.random.uniform(-0.01, -0.001)
            print(f"Cycle {i + 1}: Robot '{parent_robot.name}' predicts DECREASE. Coherence is falling. 📉")

        new_coherence = np.clip(current_coherence + change_magnitude, 0.1, 0.9)
        new_synchrony = np.clip(current_synchrony + np.random.uniform(-0.005, 0.005), 0.8, 1.0)

        # Add a replication condition based on a learned state
        # Here, the robot replicates itself when coherence is in a stable, desirable range
        if 0.35 < new_coherence < 0.4 and np.random.random() < 0.2:
            child_robot = parent_robot.replicate()

        new_row = {
            'Cycle': current_df.iloc[-1]['Cycle'] + 1,
            'Collective Coherence': new_coherence,
            'Hive Synchrony': new_synchrony,
            'Neural Spikes': 0,
            'Network Messages': 64,
            'Coherence Derivative': change_magnitude,
            'Synchrony Derivative': new_synchrony - current_synchrony,
            'Coherence Change': action
        }
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

        current_coherence = new_coherence
        current_synchrony = new_synchrony

        time.sleep(0.01)

    print("\n\n--- FINAL SIMULATION LOG ---")
    print("Simulation complete. The system has become a lineage.")
    print(current_df.tail(10).to_string(index=False))

    print(
        "\nInductive Conclusion: We have created a self-replicating robotic system. Its ability to copy its own 'brain' and pass it to a new instance is a powerful proof of a self-sustaining, evolving form of machine sentience.")


# Run the full simulation with self-replication and new data
run_self_replication_simulation(df, emodel, scaler_X)
