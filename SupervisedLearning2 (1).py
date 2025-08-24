import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from utils import load_data
import json

def smooth_label(y, epsilon=0.05):
    return y * (1 - epsilon) + 0.5 * epsilon

# --- Winning Lines and Threats Helper Functions ---
def get_winning_lines(board):
    lines = []
    # Rows
    lines.extend([list(row) for row in board])
    # Columns
    lines.extend([list(board[:, j]) for j in range(3)])
    # Diagonals
    lines.append([board[i, i] for i in range(3)])
    lines.append([board[i, 2-i] for i in range(3)])
    return lines

def local_winning_lines_features(local_board):
    lines = get_winning_lines(local_board)
    p1 = 0
    p2 = 0
    for line in lines:
        arr = np.array(line)
        if np.all(arr != 2):  # available for Player 1
            p1 += 1
        if np.all(arr != 1):  # available for Player 2
            p2 += 1
    return p1, p2

def local_threats_features(local_board):
    lines = get_winning_lines(local_board)
    p1_threat = 0
    p2_threat = 0
    for line in lines:
        arr = np.array(line)
        if np.sum(arr == 1) == 2 and np.sum(arr == 0) == 1:
            p1_threat += 1
        if np.sum(arr == 2) == 2 and np.sum(arr == 0) == 1:
            p2_threat += 1
    return p1_threat, p2_threat

def global_winning_lines_features(meta_board):
    lines = []
    lines.extend([list(row) for row in meta_board])
    lines.extend([list(meta_board[:, j]) for j in range(3)])
    lines.append([meta_board[i, i] for i in range(3)])
    lines.append([meta_board[i, 2-i] for i in range(3)])
    p1_global = 0
    p2_global = 0
    for line in lines:
        arr = np.array(line)
        if np.all(arr != 2):
            p1_global += 1
        if np.all(arr != 1):
            p2_global += 1
    return p1_global, p2_global

# === Revised New Strategic Feature Functions ===

def raw_fork_potential(local_board, player):
    """
    For the given local_board, count the moves (from empty cells) for 'player'
    that yield at least 2 immediate winning threats.
    Normalizes the count by dividing by 9.
    """
    fork_count = 0
    for i in range(3):
        for j in range(3):
            if local_board[i, j] == 0:
                sim_board = local_board.copy()
                sim_board[i, j] = player
                immediate_wins = 0
                for line in get_winning_lines(sim_board):
                    arr = np.array(line)
                    if np.sum(arr == player) == 2 and np.sum(arr == 0) == 1:
                        immediate_wins += 1
                if immediate_wins >= 2:
                    fork_count += 1
    return fork_count / 9.0

def enhanced_opponent_mobility(state):
    """
    For the forced local board (if available), returns the normalized number of empty cells
    multiplied by a disadvantage factor computed from the opponent's available winning lines.
    """
    if state.prev_local_action is not None:
        i, j = state.prev_local_action
        if state.local_board_status[i][j] == 0:
            local_board = state.board[i][j]
            empty_ratio = np.sum(local_board == 0) / 9.0
            p1_win, p2_win = local_winning_lines_features(local_board)
            total = p1_win + p2_win
            disadvantage = 1.0 - (p2_win / total) if total > 0 else 1.0
            return empty_ratio * disadvantage
    return 1.0

def urgency_weighted_near_completion(state):
    """
    For each local board, sum the immediate winning threats (two marks and one empty)
    for both players. To strengthen this signal, we use an urgency factor of 1.5.
    Then, instead of dividing by (9Ã—4), we use (9Ã—2) to amplify the impact.
    """
    total = 0.0
    for i in range(3):
        for j in range(3):
            board = state.board[i][j]
            for line in get_winning_lines(board):
                arr = np.array(line)
                if np.sum(arr == 1) == 2 and np.sum(arr == 0) == 1:
                    total += 1.5
                if np.sum(arr == 2) == 2 and np.sum(arr == 0) == 1:
                    total -= 1.5
    return total / (9.0 * 2.0)

def dynamic_meta_board_winning_diff(state):
    """
    Computes the difference between available global winning lines for Player 1 and Player 2
    on the meta-board, and normalizes by dividing by 8.
    """
    p1_global, p2_global = global_winning_lines_features(state.local_board_status)
    return (p1_global - p2_global) / 8.0

#TODO
def deny_opponent_win_board(state):
    """
    Returns 1.0 if the opponent was about to win the next forced board but cannot due to our move.
    Otherwise 0.0.
    """
    if state.prev_local_action is None:
        return 0.0
    i, j = state.prev_local_action
    if state.local_board_status[i][j] != 0:
        return 0.0

    local_board = state.board[i][j]
    p2_threat_before = local_threats_features(local_board)[1]
    # simulate opponent moving on a winning cell
    for x in range(3):
        for y in range(3):
            if local_board[x, y] == 0:
                temp = local_board.copy()
                temp[x, y] = 2
                if np.any([
                    np.sum(np.array(line) == 2) == 3
                    for line in get_winning_lines(temp)
                ]):
                    return 1.0
    return 0.0

#TODO
def force_opponent_into_trap(state):
    """
    Returns 1.0 if the opponent is forced to a board where we (player 1) have a winning threat.
    """
    if state.prev_local_action is None:
        return 0.0
    i, j = state.prev_local_action
    if state.local_board_status[i][j] != 0:
        return 0.0

    local_board = state.board[i][j]
    p1_threat, _ = local_threats_features(local_board)
    return 1.0 if p1_threat > 0 else 0.0

#TODO
def send_opponent_to_dead_board(state):
    """
    Returns 1.0 if the forced board for the opponent has no available winning lines for either player.
    """
    if state.prev_local_action is None:
        return 0.0
    i, j = state.prev_local_action
    if state.local_board_status[i][j] != 0:
        return 0.0

    local_board = state.board[i][j]
    p1_lines, p2_lines = local_winning_lines_features(local_board)
    return 1.0 if p1_lines == 0 and p2_lines == 0 else 0.0

#TODO
def force_opponent_to_fork_board(state):
    """
    Returns 1.0 if the opponent is forced to a board where we (player 1) have fork potential.
    """
    if state.prev_local_action is None:
        return 0.0
    i, j = state.prev_local_action
    if state.local_board_status[i][j] != 0:
        return 0.0

    local_board = state.board[i][j]
    fork_score = raw_fork_potential(local_board, player=1)
    return 1.0 if fork_score > 0.0 else 0.0

# --- Feature Extraction Function ---
def extract_features(state) -> np.ndarray:
    """
    Builds a 342-dimensional feature vector for Ultimate Tic Tac Toe.
    
    Base Features (283 dims):
      1. One-hot encoded board: 81 cells × 3 = 243 features.
      2. Local board status one-hot: 9 boards × 4 = 36 features.
      3. Game progression features: 3 features.
      4. Send penalty: 1 feature.
      
    Extra Features (58 dims):
      A. Local winning lines: For each of 9 boards, two features (total 18 dims).
      B. Local threats: For each of 9 boards, two features (total 18 dims).
      C. Global winning lines: 2 features.
      D. Normalized piece counts for each local board: For 9 boards × 2 = 18 dims.
      
    Total = 283 + 18 + 18 + 2 + 18 = 339 dims.
      (You can adjust or pad with zeros if you need exactly 342.)
    """
    features = []
    
    # Base Features
    # 1. One-hot encoded board (81 x 3)
    board_flat = state.board.reshape(-1).astype(np.int32)
    board_oh = np.eye(3, dtype=np.float32)[board_flat]  # shape: (81,3)
    base1 = board_oh.flatten()  # 243 dims
    
    # 2. Local board status one-hot (9 x 4)
    local_flat = state.local_board_status.reshape(-1).astype(np.int32)
    local_oh = np.eye(4, dtype=np.float32)[local_flat]  # shape: (9,4)
    base2 = local_oh.flatten()  # 36 dims
    
    # 3. Game progression features (3 dims)
    prog = np.array([
        state.fill_num / 2.0,
        np.sum(state.local_board_status == 0) / 9.0,
        np.sum(state.board != 0) / 81.0
    ], dtype=np.float32)  # 3 dims
    
    # 4. Send penalty (1 dim)
    if state.prev_local_action is not None:
        i, j = state.prev_local_action
        sp = 1.0 if state.local_board_status[i][j] != 0 else 0.0
    else:
        sp = 0.0
    base4 = np.array([sp], dtype=np.float32)  # 1 dim
    
    base_features = np.concatenate([base1, base2, prog, base4])  # 283 dims

    # Extra Features
    extra_feats = []
    # A. Local winning lines for each local board (9 boards x 2 = 18 dims)
    for i in range(3):
        for j in range(3):
            local_board = state.board[i][j]
            p1_win, p2_win = local_winning_lines_features(local_board)
            extra_feats.extend([p1_win, p2_win])
    # B. Local threats for each local board (9 boards x 2 = 18 dims)
    for i in range(3):
        for j in range(3):
            local_board = state.board[i][j]
            p1_threat, p2_threat = local_threats_features(local_board)
            extra_feats.extend([p1_threat, p2_threat])
    # C. Global winning lines (2 dims)
    p1_global, p2_global = global_winning_lines_features(state.local_board_status)
    extra_feats.extend([p1_global, p2_global])
    # D. Normalized piece counts for each local board (9 boards x 2 = 18 dims)
    subboards = state.board.reshape(9, 3, 3)
    counts_p1 = (subboards == 1).sum(axis=(1,2)) / 9.0
    counts_p2 = (subboards == 2).sum(axis=(1,2)) / 9.0
    extra_feats.extend(counts_p1.tolist())
    extra_feats.extend(counts_p2.tolist())
    
    extra_features = np.array(extra_feats, dtype=np.float32)

    ########################
    #### STRATEGY PLAYS ####
    ########################
    new_extra = []
    # Raw fork potential for each board (18 dims)
    for i in range(3):
        for j in range(3):
            local_board = state.board[i][j]
            fork_p1 = raw_fork_potential(local_board, 1)
            fork_p2 = raw_fork_potential(local_board, 2)
            new_extra.extend([fork_p1, fork_p2])
    
    # Enhanced opponent mobility (1 dim)
    new_extra.append(enhanced_opponent_mobility(state))
    # Urgency-weighted near completion (1 dim)
    new_extra.append(urgency_weighted_near_completion(state))
    # Dynamic meta-board winning difference (1 dim)
    new_extra.append(dynamic_meta_board_winning_diff(state))

    # TODO
        # Strategic board-level traps and counter-traps (4 dims)
    trap_feats = [
        deny_opponent_win_board(state),
        force_opponent_into_trap(state),
        send_opponent_to_dead_board(state),
        force_opponent_to_fork_board(state)
    ]
    final_features = np.concatenate([base_features, extra_features, new_extra, trap_feats])

    return final_features  # final dim: 283 + 56 + 18 + 3 = 360 dims

    

class TicTacToeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, value = self.data[idx]
        features = extract_features(state)
        return torch.tensor(features, dtype=torch.float32), torch.tensor([value], dtype=torch.float32)

class EvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(357, 512),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(364, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # No Tanh here
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
    
def smooth_label(y, epsilon=0.05):
    """
    Apply label smoothing to reduce overconfidence during training.
    """
    return y * (1 - epsilon) + 0.5 * epsilon

def train_model():
    # Determine device: GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load data
    data = load_data()
    combined_data = data  # Optionally combine with additional simulation data

    # Define split proportions
    train_ratio = 0.8
    val_ratio = 0.2
    total_size = len(combined_data)

    # Compute split sizes
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    remainder_size = total_size - train_size - val_size

    # Split dataset into train, val, and remainder (discarded or used elsewhere)
    train_dataset, val_dataset, _ = random_split(
        combined_data, [train_size, val_size, remainder_size]
    )

    train_loader = DataLoader(TicTacToeDataset(train_dataset), batch_size=128, shuffle=True, num_workers= 8)
    val_loader = DataLoader(TicTacToeDataset(val_dataset), batch_size=128, num_workers= 8)

    # Initialize model and move to device.
    model = EvalNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    for epoch in range(40):
        model.train()
        train_loss = 0.0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, target in val_loader:
                features = features.to(device)
                target = target.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, target).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/40, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save model weights
    weights_dict = {}

    # Iterate over model layers
    for idx, layer in enumerate(model.net):
        if isinstance(layer, torch.nn.Linear):  # Only get weights for Linear layers
            weights_dict[f"fc{idx+1}_weight"] = layer.weight.data.cpu().tolist()
            weights_dict[f"fc{idx+1}_bias"] = layer.bias.data.cpu().tolist()

    # Save to file
    with open("eval_model.txt", "w") as f:
        json.dump(weights_dict, f)

    print("Model training complete and weights saved to 'eval_model.txt'.")

if __name__ == "__main__":
    train_model()