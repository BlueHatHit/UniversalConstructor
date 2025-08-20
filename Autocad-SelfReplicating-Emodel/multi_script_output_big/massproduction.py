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


# --- CONCEPTUAL ASTEROID FACTORY DATASET (MASSIVE SCALE) ---
def generate_asteroid_data(num_cycles=20000):
    """
    Generates a synthetic, massive dataset for an asteroid factory robot.
    - Resource Density, Temperature, Radiation as inputs.
    - A binary label indicating a "Mining" action.
    - A new metric to track: Energy Consumption.
    """
    cycles = np.arange(1, num_cycles + 1)

    # Simulate a complex, fluctuating environment with noise
    resource_density = 0.5 + 0.2 * np.sin(cycles * 0.001) + np.random.normal(0, 0.05, num_cycles)
    temperature = 100 * np.sin(cycles * 0.002) + np.random.normal(0, 5, num_cycles)
    radiation = 10 * np.cos(cycles * 0.003) + np.random.normal(0, 1, num_cycles)

    # Simulate a "Mining" action policy that is dependent on resource density
    mining_action = (resource_density > 0.55).astype(int)

    # Simulate energy consumption, which fluctuates with mining activity
    energy_consumption = 0.2 + 0.5 * mining_action + np.random.normal(0, 0.05, num_cycles)

    df = pd.DataFrame({
        'Cycle': cycles,
        'Resource Density': resource_density,
        'Temperature': temperature,
        'Radiation': radiation,
        'Mining Action': mining_action,
        'Energy Consumption': energy_consumption
    })

    return df


df_asteroid = generate_asteroid_data(num_cycles=20000)


# --- OPEN-SOURCE MODEL ARCHITECTURES (LeRobot Inspired) ---
# The Parent model is a conceptual RoboticsTransformer, larger and more complex.
class RoboticsTransformer(nn.Module):
    def __init__(self, input_dim):
        super(RoboticsTransformer, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)  # Significantly larger for complex data
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)

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
        self.output_layer = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.output_layer(x)
        return x


# --- DATA PREPARATION ---
X = df_asteroid[['Resource Density', 'Temperature', 'Radiation']].values
y = df_asteroid['Mining Action'].values

scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

X_tensor = torch.from_numpy(X_normalized).float()
y_tensor = torch.from_numpy(y).long()

# --- TRAINING THE PARENT ROBOTICS TRANSFORMER ---
print("--- TRAINING THE PARENT ROBOTICS TRANSFORMER for Asteroid Factory ---")
parent_model = RoboticsTransformer(input_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parent_model.parameters(), lr=0.01)

for epoch in range(2000):  # Increased epochs for massive dataset
    outputs = parent_model(X_tensor)
    loss = criterion(outputs, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 400 == 0:
        print(f'Parent RoboticsTransformer - Epoch [{epoch + 1}/2000], Loss: {loss.item():.4f}')

print("\nParent RoboticsTransformer is trained and ready to be a factory leader.")


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
        """
        Simulates the self-replication process using distillation.
        The child robot's brain learns from the parent's predictions.
        """
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


# --- SIMULATION LOOP ---
def run_asteroid_simulation(df, parent_model, scaler, num_cycles=100):
    print("\n🌟 Initializing Asteroid Factory Simulation with Self-Replicating Robot... 🌟")

    robot_alpha = SelfReplicatingRobot(name="Alpha", model_type=RoboticsTransformer,
                                       model_state=parent_model.state_dict(), scaler=scaler)

    current_state = df.iloc[0][['Resource Density', 'Temperature', 'Radiation']].values

    for i in range(num_cycles):
        action = robot_alpha.perform_action(current_state)

        if action == 1:
            print(f"Cycle {i + 1}: Robot '{robot_alpha.name}' predicts MINING. Energy consumption rising. 🔋")
            current_state = current_state + np.random.normal(0.01, 0.005, 3)
        else:
            print(f"Cycle {i + 1}: Robot '{robot_alpha.name}' predicts NO MINING. Conserving energy. 🌿")
            current_state = current_state + np.random.normal(-0.01, 0.005, 3)

        current_state = np.clip(current_state, -1, 1)

        if i == 50:
            child_robot = robot_alpha.replicate_with_distillation(robot_alpha.model)

        time.sleep(0.01)

    print("\n\n--- FINAL SIMULATION LOG ---")
    print("Simulation complete. The parent robot has successfully replicated a child.")


# Run the full simulation
run_asteroid_simulation(df_asteroid, parent_model, scaler_X)
