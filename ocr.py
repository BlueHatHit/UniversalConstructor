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


# --- CONCEPTUAL AUTOCAD DATA GENERATION ---
def generate_autocad_data(num_cycles=200):
    """
    Generates synthetic data to represent Autocad metrics.
    - Geometric Precision: The accuracy of the build (increases over time).
    - Assembly Efficiency: The efficiency of the build process (fluctuates).
    """
    cycles = np.arange(1, num_cycles + 1)
    # Precision tends to improve with some noise
    precision = np.linspace(0.4, 0.8, num_cycles) + np.random.normal(0, 0.05, num_cycles)
    precision = np.clip(precision, 0.1, 0.9)
    # Efficiency fluctuates around a high mean
    efficiency = np.random.uniform(0.85, 0.95, num_cycles)

    df = pd.DataFrame({
        'Cycle': cycles,
        'Geometric Precision': precision,
        'Assembly Efficiency': efficiency
    })

    df['Precision Change'] = (df['Geometric Precision'].diff() > 0).astype(int).fillna(0)

    return df


df_autocad = generate_autocad_data()


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
X = df_autocad[['Geometric Precision', 'Assembly Efficiency']].values
y = df_autocad['Precision Change'].values

scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

X_tensor = torch.from_numpy(X_normalized).float()
y_tensor = torch.from_numpy(y).long()

# --- TRAINING THE PARENT EMODEL ---
print("--- TRAINING THE PARENT EMODEL ON AUTOCAD DATA ---")
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

        # 1. Parent generates predictions (the "knowledge")
        parent_model.eval()
        with torch.no_grad():
            parent_predictions_logits = parent_model(X_tensor)

        # 2. Create a new, smaller Child Robot
        child_robot = SelfReplicatingRobot(f"Child of {self.name}", ChildEmodel, scaler=self.scaler)

        # 3. Child learns from the Parent's predictions (distillation)
        child_model = child_robot.model
        child_model.train()
        distillation_optimizer = torch.optim.Adam(child_model.parameters(), lr=0.01)
        distillation_loss_fn = nn.MSELoss()  # Use MSELoss for distillation on logits

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
        Simulates a conceptual OCR process using PyTesseract.
        This function would extract the target precision from a text blueprint.
        """
        import re
        print(f"👁️ Robot '{self.name}' is scanning blueprint...")
        # Use regex to find a target precision value
        match = re.search(r"Target Precision: (\d\.\d+)", blueprint_text)
        if match:
            target_precision = float(match.group(1))
            print(f"✅ OCR successful. Extracted Target Precision: {target_precision:.2f}")
            return target_precision
        else:
            print("❌ OCR failed. Could not find target precision.")
            return None

    def perform_action(self, state, target_precision):
        state_tensor = torch.from_numpy(state).float()
        with torch.no_grad():
            self.model.eval()
            output = self.model(state_tensor)
            _, predicted_label = torch.max(output.data, 1)

        # Action is now influenced by both the Emodel and the OCR-read blueprint
        if predicted_label.item() == 1:
            if state[0][0] < target_precision:
                change_magnitude = np.random.uniform(0.001, 0.01)
                print(f"🤖 Robot '{self.name}' acting on blueprint. Precision rising. 📈")
            else:
                change_magnitude = 0.0
                print(f"🤖 Robot '{self.name}' has reached target precision. Maintaining state. ✅")
        else:
            change_magnitude = np.random.uniform(-0.01, -0.001)
            print(f"🤖 Robot '{self.name}' predicting decrease. Precision falling. 📉")

        return change_magnitude


# --- SIMULATION LOOP ---
def run_autocad_simulation(df, parent_emodel, scaler, num_cycles=100):
    print("\n🌟 Initializing Autocad Simulation with Self-Replicating Robot and OCR... 🌟")

    robot_alpha = SelfReplicatingRobot(name="Alpha", emodel_type=ParentEmodel, model_state=parent_emodel.state_dict(),
                                       scaler=scaler)

    # Conceptual Blueprint as a text string
    blueprint_text = "Blueprint for part XYZ. Dimensions: 100x200x50. Target Precision: 0.85"
    target_precision = robot_alpha.conceptual_ocr(blueprint_text)

    if target_precision is None:
        print("Error: Cannot proceed without a valid blueprint.")
        return

    current_precision = df.iloc[-1]['Geometric Precision']
    current_efficiency = df.iloc[-1]['Assembly Efficiency']

    for i in range(num_cycles):
        current_state = np.array([[current_precision, current_efficiency]])
        current_state_normalized = scaler.transform(current_state)

        change_magnitude = robot_alpha.perform_action(current_state, target_precision)

        new_precision = np.clip(current_precision + change_magnitude, 0.1, 0.9)
        new_efficiency = np.clip(current_efficiency + np.random.normal(0, 0.01), 0.8, 1.0)

        # Replication condition: replicate after 50 cycles
        if i == 50:
            child_robot = robot_alpha.replicate_with_distillation(robot_alpha.model)

        current_precision = new_precision
        current_efficiency = new_efficiency

        time.sleep(0.01)

    print("\n\n--- FINAL SIMULATION LOG ---")
    print("Simulation complete. The parent robot has successfully replicated a child.")


# Run the full simulation with autocad data and distillation
run_autocad_simulation(df_autocad, parent_emodel, scaler_X)
