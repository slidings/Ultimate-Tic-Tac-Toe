import torch
import numpy as np
import json
from utils import load_data, State
from simplifiedLearning import torch_extract_features


def precompute_features(data_path: str, save_path: str):
    data = load_data()
    features = []
    targets = []

    for state_idx, (orig_state, utility) in enumerate(data):
        if state_idx % 1000 == 0:
            print(f"Processing state {state_idx}/{len(data)}")
        features.append(torch_extract_features(orig_state))
        targets.append(utility)

    features_tensor = torch.stack(features)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    torch.save((features_tensor, targets_tensor), save_path)
    print(f"Saved {len(features)} states to {save_path}")

if __name__ == '__main__':
    precompute_features("data.pkl", "precomputed_features.pt")
