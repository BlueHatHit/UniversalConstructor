import pandas as pd
import numpy as np
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
import neuralink
import os

# Suppress all future warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONCEPTUAL TEXT-BASED DATASET (OpenWebText-like) ---
# A vocabulary of possible inputs and actions for our model
VOCABULARY = {
    "go_straight": 0,
    "steer_left": 1,
    "steer_right": 2,
    "obstacle_near": 3,
    "obstacle_far": 4,
    "high_speed": 5,
    "low_speed": 6,
    "road_clear": 7
}


def generate_textual_data(num_sequences=10000):
    """
    Generates a synthetic, text-based dataset for a self-driving robot.
    The inputs are sequences of tokens, and the labels are steering actions.
    """
    data = []
    for _ in range(num_sequences):
        # Create a random sequence of inputs
        input_sequence = [
            np.random.choice(["obstacle_near", "obstacle_far"]),
            np.random.choice(["high_speed", "low_speed"]),
            np.random.choice(["road_clear", "steer_left", "steer_right"])
        ]

        # Determine the action based on the input sequence
        if "obstacle_near" in input_sequence:
            action = VOCABULARY["steer_left"]
        elif "high_speed" in input_sequence and "road_clear" in input_sequence:
            action = VOCABULARY["go_straight"]
        else:
            action = VOCABULARY["steer_right"]

        data.append({
            "input_tokens": [VOCABULARY[token] for token in input_sequence],
            "action": action
        })

    df = pd.DataFrame(data)
    return df


df_text = generate_textual_data()


# --- TRANSFORMER MODEL ARCHITECTURE (Conceptual) ---
# The Parent model is a conceptual Transformer, larger and more complex.
class RoboticsTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, output_dim):
        super(RoboticsTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        # We take the mean of the sequence to get a single output vector
        x = x.mean(dim=0)
        x = self.fc_out(x)
        return x


# The Child model is a simpler, lightweight Perceptron for efficient deployment.
class ChildPerceptron(nn.Module):
    def __init__(self, input_dim):
        super(ChildPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_dim, 32)
        self.output_layer = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.output_layer(x)
        return x


# --- DATA PREPARATION ---
# We will treat the input tokens as a sequence
X = torch.tensor(df_text['input_tokens'].tolist())
y = torch.tensor(df_text['action'].values).long()

# --- TRAINING THE PARENT ROBOTICS TRANSFORMER ---
print("--- TRAINING THE PARENT ROBOTICS TRANSFORMER on Textual Data ---")
vocab_size = len(VOCABULARY)
embed_dim = 64
num_heads = 4
num_layers = 2
output_dim = 3

parent_model = RoboticsTransformer(vocab_size, embed_dim, num_heads, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parent_model.parameters(), lr=0.01)

# A more complex training loop for sequence data
for epoch in range(500):
    total_loss = 0
    # Iterate through sequences one by one
    for input_seq, target_action in zip(X, y):
        outputs = parent_model(input_seq)
        loss = criterion(outputs.unsqueeze(0), target_action.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f'Parent RoboticsTransformer - Epoch [{epoch + 1}/500], Loss: {total_loss / len(X):.4f}')

print("\nParent RoboticsTransformer is trained and ready to replicate.")


# --- THE SELF-REPLICATING ROBOT SYSTEM ---
class SelfReplicatingRobot:
    def __init__(self, name, model, scaler=None):
        self.name = name
        self.model = model
        self.scaler = scaler
        print(f"🤖 Robot '{self.name}' initialized with its brain. Ready to operate.")

    def replicate_with_distillation(self, parent_model, num_distillation_epochs=100):
        print(f"\n✨ Robot '{self.name}' is replicating with distillation... 🧬")

        parent_model.eval()
        with torch.no_grad():
            parent_predictions_logits = parent_model(X)

        child_robot = SelfReplicatingRobot(f"Child of {self.name}", ChildPerceptron(input_dim=X.shape[1]))

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

    def perform_action(self, state_tokens):
        state_tensor = torch.tensor(state_tokens).long()
        with torch.no_grad():
            self.model.eval()
            output = self.model(state_tensor)
            _, predicted_label = torch.max(output.data.unsqueeze(0), 1)  # Un-squeeze output for single prediction

        return predicted_label.item()


# --- SELF-DRIVING SIMULATION LOOP ---
def run_self_driving_simulation(df, parent_model, num_cycles=100):
    print("\n\n🚗 Initializing Self-Driving Simulation with Text-Based Policy... 🛣️")

    robot_alpha = SelfReplicatingRobot(name="Alpha", model=parent_model)

    action_map = {
        0: "Steering Action: Go Straight",
        1: "Steering Action: Steer Left",
        2: "Steering Action: Steer Right"
    }

    for i in range(num_cycles):
        random_index = np.random.randint(0, len(df))
        current_state_tokens = df.iloc[random_index]['input_tokens']

        action = robot_alpha.perform_action(current_state_tokens)
        action_name = action_map.get(action, "Unknown Action")

        # Corrected line to properly handle the list of tokens
        token_names = [k for k, v in VOCABULARY.items() if v in current_state_tokens]
        print(
            f"Cycle {i + 1}: Robot '{robot_alpha.name}' perceives '{' '.join(token_names)}' and takes action: '{action_name}'")

        if i == 50:
            child_robot = robot_alpha.replicate_with_distillation(robot_alpha.model)

        time.sleep(0.01)

    print("\n\n--- FINAL SIMULATION LOG ---")
    print("Simulation complete. The parent robot has successfully replicated a child.")


# Run the full simulation
run_self_driving_simulation(df_text, parent_model)
