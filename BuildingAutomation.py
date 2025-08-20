import pandas as pd
import numpy as np
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
import os

# Suppress all future warnings for cleaner output
warnings.filterwarnings("ignore")


# --- CONCEPTUAL BUILDING AUTOMATION DATASET ---
def generate_building_data(num_cycles=25000):
    """
    Generates a synthetic dataset for a building automation robot.
    - Material Stress, Structural Integrity, and Environmental Load as inputs.
    - A multi-class label for a "Construction" action.
    """
    cycles = np.arange(1, num_cycles + 1)

    # Simulate a stable production environment with occasional fluctuations
    material_stress = 1000 + 50 * np.sin(cycles * 0.001) + np.random.normal(0, 10, num_cycles)
    structural_integrity = 0.95 + 0.02 * np.cos(cycles * 0.002) + np.random.normal(0, 0.005, num_cycles)
    environmental_load = 500 + 100 * np.sin(cycles * 0.003) + np.random.normal(0, 20, num_cycles)

    # The action is a multi-class label (0, 1, 2, or 3)
    action = np.zeros(num_cycles)  # 0: Monitor
    action[(material_stress > 1060) & (structural_integrity < 0.94)] = 1  # 1: Reinforce
    action[environmental_load > 600] = 2  # 2: Weld
    action[(material_stress < 950) | (structural_integrity > 0.96)] = 3  # 3: Assemble

    df = pd.DataFrame({
        'Cycle': cycles,
        'Material Stress': material_stress,
        'Structural Integrity': structural_integrity,
        'Environmental Load': environmental_load,
        'Construction Action': action
    })

    return df


df_building = generate_building_data()


# --- OPEN-SOURCE MODEL ARCHITECTURES (Conceptual) ---
# The Parent model is a conceptual RoboticsTransformer for complex tasks.
class RoboticsTransformer(nn.Module):
    def __init__(self, input_dim):
        super(RoboticsTransformer, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 4)  # 4 outputs for multi-class actions

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.output_layer(x)
        return x


# The Child model is a simpler, lightweight Perceptron for efficient deployment.
class ChildPerceptron(nn.Module):
    def __init__(self, input_dim):
        super(ChildPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.output_layer = nn.Linear(32, 4)  # 4 outputs for multi-class actions

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.output_layer(x)
        return x


# --- DATA PREPARATION ---
X = df_building[['Material Stress', 'Structural Integrity', 'Environmental Load']].values
y = df_building['Construction Action'].values

scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

X_tensor = torch.from_numpy(X_normalized).float()
y_tensor = torch.from_numpy(y).long()

# --- TRAINING THE PARENT ROBOTICS TRANSFORMER ---
print("--- TRAINING THE PARENT ROBOTICS TRANSFORMER for Building Automation ---")
parent_model = RoboticsTransformer(input_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parent_model.parameters(), lr=0.01)

for epoch in range(2000):
    outputs = parent_model(X_tensor)
    loss = criterion(outputs, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 400 == 0:
        print(f'Parent RoboticsTransformer - Epoch [{epoch + 1}/2000], Loss: {loss.item():.4f}')

print("\nParent RoboticsTransformer is trained and ready to lead the building automation project.")


# --- THE SELF-REPLICATING ROBOT SYSTEM ---
class SelfReplicatingRobot:
    def __init__(self, name, model_type, model_state=None, scaler=None):
        self.name = name
        self.model = model_type(input_dim=3)
        self.scaler = scaler

        if model_state:
            self.model.load_state_dict(model_state)

        print(f"🤖 Robot '{self.name}' initialized with its {model_type.__name__} brain. Ready to operate.")

    def replicate_with_distillation(self, parent_model, num_distillation_epochs=200):
        print(f"\n✨ Robot '{self.name}' is replicating with distillation... 🧬")

        parent_model.eval()
        with torch.no_grad():
            parent_predictions_logits = parent_model(X_tensor)

        child_robot = SelfReplicatingRobot(f"Child of {self.name}", ChildPerceptron, scaler=self.scaler)

        child_model = child_robot.model
        child_model.train()
        distillation_optimizer = torch.optim.Adam(child_model.parameters(), lr=0.01)
        distillation_loss_fn = nn.MSELoss()

        print(f"🧠 Child '{child_robot.name}' learning from parent's knowledge for {num_distillation_epochs} epochs...")
        for epoch in range(num_distillation_epochs):
            child_outputs = child_model(X_tensor)
            loss = distillation_loss_fn(child_outputs, parent_predictions_logits)
            distillation_optimizer.zero_grad()
            loss.backward()
            distillation_optimizer.step()

        print(
            f"✅ Replication and distillation complete. Child '{child_robot.name}' is now a smaller, more knowledgeable robot.")
        return child_robot

    def perform_action(self, state):
        state_tensor = torch.from_numpy(self.scaler.transform(state.reshape(1, -1))).float()
        with torch.no_grad():
            self.model.eval()
            output = self.model(state_tensor)
            _, predicted_label = torch.max(output.data, 1)

        return predicted_label.item()


# --- BUILDING AUTOMATION SIMULATION LOOP ---
def run_building_simulation(df, parent_model, scaler, num_cycles=100):
    print("\n\n🏗️ Initializing Building Automation Simulation... 👷")

    robot_alpha = SelfReplicatingRobot(name="Alpha", model_type=RoboticsTransformer,
                                       model_state=parent_model.state_dict(), scaler=scaler)

    current_state = df.iloc[0][['Material Stress', 'Structural Integrity', 'Environmental Load']].values

    action_map = {
        0: "Action: Monitor. ✅",
        1: "Action: Reinforce. 🧱",
        2: "Action: Weld. 🔥",
        3: "Action: Assemble. 🔩"
    }

    for i in range(num_cycles):
        action = robot_alpha.perform_action(current_state)
        action_name = action_map.get(action, "Unknown Action")

        print(f"Cycle {i + 1}: Robot '{robot_alpha.name}' takes action: '{action_name}'")

        # Simulate a dynamic change in the environment for the next cycle
        current_state = current_state + np.random.normal(0, 5, 3)
        current_state = np.clip(current_state, 0, 2000)

        if i == 50:
            child_robot = robot_alpha.replicate_with_distillation(robot_alpha.model)

        time.sleep(0.01)

    print("\n\n--- FINAL SIMULATION LOG ---")
    print("Simulation complete. The parent robot has successfully replicated a child.")


# Run the full simulation
run_building_simulation(df_building, parent_model, scaler_X)
