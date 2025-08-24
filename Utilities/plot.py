import matplotlib.pyplot as plt
import torch
import json
import numpy as np

from manualAgent import StudentAgent
from utils import State, load_data, convert_board_to_string

data = load_data()
student_agent = StudentAgent()

true_labels = []
predicted_values = []
max_loss_index = 0
max_loss = -np.inf

# Define the evaluation subset
start_idx = 50
end_idx = 500
subset = data[start_idx:end_idx]

for index, (state, label) in enumerate(subset):
    predicted = student_agent.evaluate(state)
    loss = (predicted - label) ** 2

    if loss > max_loss:
        max_loss = loss
        max_loss_index = index

    predicted_values.append(predicted)
    true_labels.append(label)

# Get worst performing data point from the subset
worst_state, worst_label = subset[max_loss_index]
worst_prediction = predicted_values[max_loss_index]

print(f"\nWorst Data Point (Index {start_idx + max_loss_index}):")
print(f"True Label: {worst_label:.4f}")
print(f"Predicted Value: {worst_prediction:.4f}")
print(f"Squared Error: {max_loss:.4f}")

# Print board state details
print("\nBoard State Analysis:")
print("Local Board Status (Meta Board):")
print(convert_board_to_string(worst_state.board))
print("\nMeta Board Status:")
print(worst_state.local_board_status)
print("\nCurrent Player:", worst_state.fill_num)
print("Previous Local Action:", worst_state.prev_local_action)

# Compute MSE and plot
true_labels_np = np.array(true_labels)
predicted_values_np = np.array(predicted_values)
mse = np.mean((true_labels_np - predicted_values_np) ** 2)
print(f"\nMean Squared Error (MSE): {mse:.4f}")

plt.figure(figsize=(10, 5))
plt.scatter(true_labels, predicted_values, alpha=0.5)
plt.plot([-1, 1], [-1, 1], 'r--', label='Ideal')
plt.xlabel("True Utility")
plt.ylabel("Predicted Utility")
plt.title(f"Evaluation Performance (MSE: {mse:.4f})")
plt.grid(True)
plt.show()