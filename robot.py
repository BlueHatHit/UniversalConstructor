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


# --- CONCEPTUAL 3D PRINTER DATA GENERATION ---
def generate_3d_printer_data(num_cycles=200):
    """
    Generates synthetic data to represent 3D printer metrics.
    - Layer Adhesion: Quality of the print (increases over time).
    - Print Speed: Efficiency of the print process (fluctuates).
    """
    cycles = np.arange(1, num_cycles + 1)
    # Layer adhesion tends to improve with some noise
    adhesion = np.linspace(0.6, 0.9, num_cycles) + np.random.normal(0, 0.03, num_cycles)
    adhesion = np.clip(adhesion, 0.1, 1.0)
    # Print speed fluctuates around a high mean
    speed = np.random.uniform(100, 150, num_cycles)

    df = pd.DataFrame({
        'Cycle': cycles,
        'Layer Adhesion': adhesion,
        'Print Speed': speed
    })

    df['Adhesion Change'] = (df['Layer Adhesion'].diff() > 0).astype(int).fillna(0)

    return df


df_3d_printer = generate_3d_printer_data()


# --- MODEL DEFINITION: PARENT (TEACHER) AND CHILD (STUDENT) ---
# The Parent Emodel is more complex, acting as the knowledge source.
class ParentEmodel(nn.Module):
    def __init__(self, input_dim):
        super(ParentEmodel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


# The Child Emodel is a simpler, smaller version to simulate a new robot.
class ChildEmodel(nn.Module):
    def __init__(self, input_dim):
        super(ChildEmodel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 8)
        self.output_layer = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.output_layer(x)
        return x


# --- DATA PREPARATION ---
X = df_3d_printer[['Layer Adhesion', 'Print Speed']].values
y = df_3d_printer['Adhesion Change'].values

scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

X_tensor = torch.from_numpy(X_normalized).float()
y_tensor = torch.from_numpy(y).long()

# --- TRAINING THE PARENT EMODEL ---
print("--- TRAINING THE PARENT EMODEL ON 3D PRINTER DATA ---")
parent_emodel = ParentEmodel(input_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parent_emodel.parameters(), lr=0.01)

for epoch in range(500):
    outputs = parent_emodel(X_tensor)
    loss = criterion(outputs, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Parent Emodel - Epoch [{epoch + 1}/500], Loss: {loss.item():.4f}')

print("\nParent Emodel is trained and ready to replicate its knowledge.")


# --- THE SELF-REPLICATING ROBOT SYSTEM ---
class SelfReplicatingRobot:
    def __init__(self, name, emodel_type, model_state=None, scaler=None):
        self.name = name
        self.model = emodel_type(input_dim=2)
        self.scaler = scaler

        if model_state:
            self.model.load_state_dict(model_state)

        print(f"🤖 Robot '{self.name}' initialized with its Emodel brain. Ready to operate.")

    def replicate_with_distillation(self, parent_model, num_distillation_epochs=50):
        """
        Simulates the self-replication process using distillation.
        The child robot's brain learns from the parent's predictions.
        """
        print(f"\n✨ Robot '{self.name}' is replicating with distillation... 🧬")

        parent_model.eval()
        with torch.no_grad():
            parent_predictions_logits = parent_model(X_tensor)

        child_robot = SelfReplicatingRobot(f"Child of {self.name}", ChildEmodel, scaler=self.scaler)

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
            f"✅ Replication and distillation complete. Child '{child_robot.name}' is now a smaller, knowledgeable robot.")
        return child_robot

    def conceptual_ocr(self, blueprint_text):
        """
        Simulates a conceptual OCR process to read a 3D printer blueprint.
        """
        import re
        print(f"👁️ Robot '{self.name}' is scanning blueprint...")
        match = re.search(r"Target Layer Adhesion: (\d\.\d+)", blueprint_text)
        if match:
            target_adhesion = float(match.group(1))
            print(f"✅ OCR successful. Extracted Target Layer Adhesion: {target_adhesion:.2f}")
            return target_adhesion
        else:
            print("❌ OCR failed. Could not find target layer adhesion.")
            return None

    def perform_action(self, state, target_adhesion, environmental_noise):
        """
        Uses the Emodel to predict the next action, but also adapts to noise.
        """
        # The robot's state now includes the perceived environmental noise
        perceived_state = np.array([[state[0][0], state[0][1] + environmental_noise]])
        state_tensor = torch.from_numpy(self.scaler.transform(perceived_state)).float()

        with torch.no_grad():
            self.model.eval()
            output = self.model(state_tensor)
            _, predicted_label = torch.max(output.data, 1)

        if predicted_label.item() == 1:
            if state[0][0] < target_adhesion:
                # Robot adjusts its action to counteract the noise
                change_magnitude = np.random.uniform(0.001, 0.01) + abs(environmental_noise) * 0.5
                print(f"🤖 Robot '{self.name}' acting on blueprint. Layer adhesion rising. 📈")
            else:
                change_magnitude = 0.0
                print(f"🤖 Robot '{self.name}' has reached target adhesion. Maintaining state. ✅")
        else:
            change_magnitude = np.random.uniform(-0.01, -0.001)
            print(f"🤖 Robot '{self.name}' predicting decrease. Layer adhesion falling. 📉")

        return change_magnitude


# --- SIMULATION LOOP ---
def run_3d_printer_simulation(df, parent_emodel, scaler, num_cycles=100):
    print("\n🌟 Initializing 3D Printer Simulation with Self-Replicating Robot and OCR... 🌟")

    robot_alpha = SelfReplicatingRobot(name="Alpha", emodel_type=ParentEmodel, model_state=parent_emodel.state_dict(),
                                       scaler=scaler)

    blueprint_text = "3D Print Job: Hexagon Lattice. Material: PLA. Target Layer Adhesion: 0.90"
    target_adhesion = robot_alpha.conceptual_ocr(blueprint_text)

    if target_adhesion is None:
        print("Error: Cannot proceed without a valid blueprint.")
        return

    current_adhesion = df.iloc[-1]['Layer Adhesion']
    current_speed = df.iloc[-1]['Print Speed']

    for i in range(num_cycles):
        # Introduce conceptual real-world noise
        environmental_noise = np.random.normal(0, 0.01)

        current_state = np.array([[current_adhesion, current_speed]])

        change_magnitude = robot_alpha.perform_action(current_state, target_adhesion, environmental_noise)

        new_adhesion = np.clip(current_adhesion + change_magnitude, 0.1, 1.0)
        new_speed = np.clip(current_speed + np.random.normal(0, 1), 100, 150)

        if i == 50:
            child_robot = robot_alpha.replicate_with_distillation(robot_alpha.model)

        current_adhesion = new_adhesion
        current_speed = new_speed

        time.sleep(0.01)

    print("\n\n--- FINAL SIMULATION LOG ---")
    print("Simulation complete. The parent robot has successfully replicated a child.")


# Run the full simulation with 3D printer data and distillation
run_3d_printer_simulation(df_3d_printer, parent_emodel, scaler_X)
