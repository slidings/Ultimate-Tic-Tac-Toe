import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from utils import load_data, convert_board_to_string
import json
import datetime
import numpy as np

# First 2 represents which sub-board, last 2 represents pos within sub-board
Action = tuple[int, int, int, int]

# Specifies move's location within sub-board
LocalBoardAction = tuple[int, int]

WINNING_LINE_INDICES = torch.tensor([
    [0, 1, 2],  # Row 1
    [3, 4, 5],  # Row 2
    [6, 7, 8],  # Row 3
    [0, 3, 6],  # Column 1
    [1, 4, 7],  # Column 2
    [2, 5, 8],  # Column 3
    [0, 4, 8],  # Diagonal
    [2, 4, 6]   # Diagonal
], dtype=torch.long)

# Generate winning lines
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

# Potential winning local lines
def local_winning_lines_features(local_board):
    board_flat = local_board.flatten()
    lines = board_flat[WINNING_LINE_INDICES]
    p1 = np.sum(np.all(lines != 2, axis=1))
    p2 = np.sum(np.all(lines != 1, axis=1))
    return p1, p2

def free_lines_features(board: torch.Tensor):
    board_flat = board.view(-1)
    lines = board_flat[WINNING_LINE_INDICES]
    is_free_line = torch.all(lines == 0, dim=1)
    count = torch.sum(is_free_line).float()
    return count

# Potential winning global lines (Includes filter for 3 drawn board)
def global_winning_lines_features(meta_board):
    board_flat = meta_board.flatten()
    lines = board_flat[WINNING_LINE_INDICES]
    p1_global = np.sum(np.all(lines != 2 & (lines != 3), axis=1))
    p2_global = np.sum(np.all(lines != 1 & (lines != 3), axis=1))
    return p1_global, p2_global

def local_threats_features(local_board):
    board_flat = local_board.flatten()
    lines = board_flat[WINNING_LINE_INDICES]
    p1_threat = np.sum((np.sum(lines == 1, axis=1) == 2) & (np.sum(lines == 0, axis=1) == 1))
    p2_threat = np.sum((np.sum(lines == 2, axis=1) == 2) & (np.sum(lines == 0, axis=1) == 1))
    return p1_threat, p2_threat

def global_threats_features(global_board):
    board_flat = global_board.flatten()
    lines = board_flat[WINNING_LINE_INDICES]
    p1_threat = np.sum((np.sum(lines == 1, axis=1) == 2) & (np.sum(lines == 0, axis=1) == 1))
    p2_threat = np.sum((np.sum(lines == 2, axis=1) == 2) & (np.sum(lines == 0, axis=1) == 1))
    return p1_threat, p2_threat


def torch_local_winning_lines_features(local_board: torch.Tensor):
    board_flat = local_board.view(-1)
    lines = board_flat[WINNING_LINE_INDICES]
    p1 = torch.sum(torch.all(lines != 2, dim=1)).float()
    p2 = torch.sum(torch.all(lines != 1, dim=1)).float()
    return p1, p2

def torch_global_winning_lines_features(meta_board: torch.Tensor):
    board_flat = meta_board.view(-1)
    lines = board_flat[WINNING_LINE_INDICES]
    p1_global = torch.sum(torch.all((lines != 2) & (lines != 3), dim=1)).float()
    p2_global = torch.sum(torch.all((lines != 1) & (lines != 3), dim=1)).float()
    return p1_global, p2_global

def torch_local_threats_features(local_board: torch.Tensor):
    board_flat = local_board.view(-1)
    lines = board_flat[WINNING_LINE_INDICES]
    p1_threat = torch.sum((torch.sum(lines == 1, dim=1) == 2) & (torch.sum(lines == 0, dim=1) == 1)).float()
    p2_threat = torch.sum((torch.sum(lines == 2, dim=1) == 2) & (torch.sum(lines == 0, dim=1) == 1)).float()
    return p1_threat, p2_threat

def torch_global_threats_features(global_board):
    board_flat = global_board.view(-1)
    lines = board_flat[WINNING_LINE_INDICES]
    p1_threat = torch.sum((torch.sum(lines == 1, dim=1) == 2) & (torch.sum(lines == 0, dim=1) == 1)).float()
    p2_threat = torch.sum((torch.sum(lines == 2, dim=1) == 2) & (torch.sum(lines == 0, dim=1) == 1)).float()
    return p1_threat, p2_threat

# For both global and local
def torch_weighted_piece_diff(board: torch.Tensor):
    # Create weight matrix as a tensor (3x3)
    weights = torch.tensor([[3.0, 2.0, 3.0],
                            [2.0, 4.0, 2.0],
                            [3.0, 2.0, 3.0]], dtype=board.dtype, device=board.device)
    total_weight = 16 # Maximum can only get 16 without winning the board entirely
    # Compute weighted sums for player 1 and 2
    p1_weight = torch.sum((board == 1).float() * weights)
    p2_weight = torch.sum((board == 2).float() * weights)
    diff = (p1_weight - p2_weight) / total_weight
    return diff

# Purely number of meta boards won
def torch_meta_board_occupation(meta_board: torch.Tensor):
    p1_won = torch.sum(meta_board == 1).float()
    p2_won = torch.sum(meta_board == 2).float()
    return (p1_won - p2_won) / 9.0

# Return number of fork by each player
def raw_fork_potential(local_board):
    p1_fork = 0
    p2_fork = 0

    p1, p2 = local_threats_features(local_board)
    if p1 > 1:
        p1_fork = p1 - 1
    if p2 > 1:
        p2_fork = p2 - 1

    return p1_fork, p2_fork

def enhanced_opponent_mobility(state):
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
    total = 0.0
    # Reshape to (9, 3, 3) to process all local boards.
    boards = state.board.reshape(9, 3, 3)
    for board in boards:
        board_flat = board.flatten()
        lines = board_flat[WINNING_LINE_INDICES]
        # Compute for player 1 and player 2 in vectorized form.
        p1_threats = np.sum((np.sum(lines == 1, axis=1) == 2) & (np.sum(lines == 0, axis=1) == 1))
        p2_threats = np.sum((np.sum(lines == 2, axis=1) == 2) & (np.sum(lines == 0, axis=1) == 1))
        total += (p1_threats * 1.5) - (p2_threats * 1.5)
    return total / (9.0 * 2.0)

def dynamic_meta_board_winning_diff(state):
    p1_global, p2_global = global_winning_lines_features(state.local_board_status)
    return (p1_global - p2_global) / 8.0

def dynamic_meta_board_winning_threats(state):
    p1_global, p2_global = global_threats_features(state.local_board_status)
    return (p1_global - p2_global) / 5.0

def force_opponent_into_trap(state):
    if state.prev_local_action is None:
        return 0.0
    
    i, j = state.prev_local_action
    if state.local_board_status[i][j] != 0:
        return 0.0

    local_board = state.board[i][j]
    p1_traps, p2_traps = local_threats_features(local_board)
    return (p1_traps - p2_traps) / 8.0

def send_opponent_to_dead_board(state):
    if state.prev_local_action is None:
        # Starting move will never have dead board
        return 0.0
    
    i, j = state.prev_local_action

    if state.local_board_status[i][j] != 0:
        # The board has ended
        return 0.0

    local_board = state.board[i][j]
    p1_lines, p2_lines = local_winning_lines_features(local_board)
    
    return 1.0 if p1_lines == 0 and p2_lines == 0 else 0.0

# Return the amount of fork potential we send our opponents to
def force_opponent_to_fork_board(state):
    if state.prev_local_action is None:
        # Impossible to have forks in the starting move
        return 0.0
    
    i, j = state.prev_local_action
    if state.local_board_status[i][j] != 0:
        if state.local_board_status[i][j] == 1:
            # We won the previous board
            return 0.1
        if state.local_board_status[i][j] == 3:
            # Slight penalty for drawing a board
            return -0.1

    local_board = state.board[i][j]
    p1_fork, _ = raw_fork_potential(local_board)

    return p1_fork / 4.0

def center_board_control(state):
    center_board = state.board[1][1]
    hyperparameter = 0.8

    score = 0.0
    if state.local_board_status[1][1] == 0:
        # Ongoing
        # Go for winning lines + near wins
        p1_threats, p2_threats = local_threats_features(center_board)
        threats_score = (p1_threats - p2_threats) / 5.0

        p1_win, p2_win = local_winning_lines_features(center_board)
        winning_potential_score = (p1_win - p2_win) / 8.0

        score = threats_score * hyperparameter + winning_potential_score * (1 - hyperparameter)

    elif state.local_board_status[1][1] == 1:
        score = 1.0
    elif state.local_board_status[1][1] == 2:
        score = -1.0
    else:
        score = 0.0
    return score

def corner_board_control(state):
    # List of coordinates for the four corner boards (meta board)
    corner_coords = [(0, 0), (0, 2), (2, 0), (2, 2)]
    # Hyperparameter for weighting between threats and winning potential
    hyperparameter = 0.8
    total_score = 0.0
    # Loop through each corner position and compute the score
    for (i, j) in corner_coords:
        corner_board = state.board[i][j]
        status = state.local_board_status[i][j]  # status of the corner board
        
        if status == 0:  # Ongoing game in this corner
            p1_threats, p2_threats = local_threats_features(corner_board)
            threats_score = (p1_threats - p2_threats) / 5.0

            p1_win, p2_win = local_winning_lines_features(corner_board)
            winning_potential_score = (p1_win - p2_win) / 8.0

            # Weighted score between threats and winning potential
            score = threats_score * hyperparameter + winning_potential_score * (1 - hyperparameter)

        elif status == 1:  # Player 1 wins this corner
            score = 1.0

        elif status == 2:  # Player 2 wins this corner
            score = -1.0

        else:  # Draw (status == 3)
            score = 0.0

        total_score += score
    # Return the average score for all corner boards
    return total_score / 4

def free_move_advantage(state):
    prev_move = state.prev_local_action
    if prev_move is None:
        # First move
        return 1.0
    else:
        i, j = prev_move
        # Game has concluded
        if state.local_board_status[i][j] != 0:
            return 1.0
        else:
            # Still on going
            return 0.0

def raw_opponent_mobility(state):
    if state.prev_local_action is None:
        return 0.0
    i, j = state.prev_local_action
    if state.local_board_status[i][j] != 0:
        return 0.0
    local_board = state.board[i][j]
    empty_cells = (local_board == 0).sum().item()
    return empty_cells / 9.0

# Extractor of all features
def torch_extract_features(state) -> torch.Tensor:
# Base Features
#     Commented out, may not be good to dilute the features
    board_flat = state.board.reshape(-1).astype(np.int32)
    board_oh = np.eye(3, dtype=np.float32)[board_flat]
    base1 = torch.from_numpy(board_oh.flatten()) * 0.3 # Convert to tensor (243 dims), smaller multiplier

    local_flat = state.local_board_status.reshape(-1).astype(np.int32)
    local_oh = np.eye(4, dtype=np.float32)[local_flat]
    base2 = torch.from_numpy(local_oh.flatten()) # Convert to tensor (36 dims)

    board = torch.from_numpy(state.board).float()
    local_board_status = torch.from_numpy(state.local_board_status)

    # Progress of the game
    prog = torch.tensor([
        state.fill_num / 2.0,
        (state.local_board_status == 0).sum().item() / 9.0,
        (state.local_board_status == 3).sum().item() / 9.0,
        (state.board != 0).sum().item() / 81.0
    ], dtype=torch.float32, device=board.device)

    # Free move checker
    if state.prev_local_action is not None:
        i, j = state.prev_local_action
        sp = 1.0 if state.local_board_status[i][j] != 0 else 0.0
    else:
        sp = 0.0
    base4 = torch.tensor([sp], dtype=torch.float32, device=board.device)

    #base1 = base1.to(board.device)
    base2 = base2.to(board.device)
    
    # Concatenate all features into a single tensor
    base_features = torch.cat([base1, base2, prog, base4])  # dim: 243 + 36 + 4 + 1 = 284

#################################################################################################
# Extra Features
    # Process 9 local boards: reshape board into (9, 3, 3)
    subboards = board.view(9, 3, 3)
    extra_feats = []
    # A. Local winning lines for each local board (9 x 2 dim)
    for board_i in subboards:
        p1_win, p2_win = torch_local_winning_lines_features(board_i)
        extra_feats.append(p1_win)
        extra_feats.append(p2_win)
    # B. Local threats for each local board (9 x 2 dim)
    for board_i in subboards:
        p1_threat, p2_threat = torch_local_threats_features(board_i)
        extra_feats.append(p1_threat)
        extra_feats.append(p2_threat)
    # C. Global winning lines for meta board (2 dim)
    p1_global, p2_global = torch_global_winning_lines_features(local_board_status)
    extra_feats.append(p1_global)
    extra_feats.append(p2_global)
    # D. Global threats  (2 dim) 
    p1_threat, p2_threat = torch_global_threats_features(board_i)
    extra_feats.append(p1_threat)
    extra_feats.append(p2_threat)
    # E. Global free lines availability (1 dim)
    extra_feats.append(free_lines_features(local_board_status))
    # F. Local free lines availability (9 dim) 
    for board_i in subboards:
        extra_feats.append(free_lines_features(board_i))
    # G. Weighted piece difference for each local board (9 dims)
    for board_i in subboards:
        diff = torch_weighted_piece_diff(board_i)
        extra_feats.append(diff)
    # H. Weighted piece difference for each meta board (1 dim)
    global_diff = torch_weighted_piece_diff(local_board_status)
    extra_feats.append(global_diff)            # dim: 18 + 18 + 2 + 2 + 9 + 1 + 9 + 1 = 60
    extra_features = torch.stack(extra_feats)  # shape: (number_of_extra_feats,)

#################################################################################################
# Strategic features
    strategy_feats = []
    # Raw fork potential for global board (2 dim)
    fork_p1, fork_p2 = raw_fork_potential(local_board_status.cpu().numpy())
    strategy_feats.append(torch.tensor(fork_p1, dtype=torch.float32, device=board.device))
    strategy_feats.append(torch.tensor(fork_p2, dtype=torch.float32, device=board.device))

    # Raw fork potential for each local board (18 dim)
    for board_i in subboards:
        fork_p1, fork_p2 = raw_fork_potential(board_i.cpu().numpy())
        strategy_feats.append(torch.tensor(fork_p1, dtype=torch.float32, device=board.device))
        strategy_feats.append(torch.tensor(fork_p2, dtype=torch.float32, device=board.device))
    
    # (9 dim)
    # Meta board occupation
    strategy_feats.append(torch_meta_board_occupation(local_board_status).clone().detach().to(dtype=torch.float32, device=board.device))
    # Raw opponent mobility
    strategy_feats.append(torch.tensor(raw_opponent_mobility(state), dtype=torch.float32, device=board.device))
    # Enhanced opponent mobility
    strategy_feats.append(torch.tensor(enhanced_opponent_mobility(state), dtype=torch.float32, device=board.device))
    # Urgency weighted near completion
    strategy_feats.append(torch.tensor(urgency_weighted_near_completion(state), dtype=torch.float32, device=board.device))
    # Dynamic meta-board winning potential diff
    strategy_feats.append(torch.tensor(dynamic_meta_board_winning_diff(state), dtype=torch.float32, device=board.device))
    # Dynamic meta_board winning threats diff
    strategy_feats.append(torch.tensor(dynamic_meta_board_winning_threats(state), dtype=torch.float32, device=board.device))
    # Center board control
    strategy_feats.append(torch.tensor(center_board_control(state), dtype=torch.float32, device=board.device))
    # Center board control
    strategy_feats.append(torch.tensor(corner_board_control(state), dtype=torch.float32, device=board.device))
    # Free move advantage
    strategy_feats.append(torch.tensor(free_move_advantage(state), dtype=torch.float32, device=board.device))

    strategy_features = torch.stack(strategy_feats) # Dim: 29

    # Strategic traps (3 dims)
    trap_feats = torch.tensor([
        force_opponent_into_trap(state),
        send_opponent_to_dead_board(state),
        force_opponent_to_fork_board(state)
    ], dtype=torch.float32, device=board.device)

    # 284 + 50 + 29 + 3 = 366
    final_features = torch.cat([base_features, extra_features, strategy_features, trap_feats])
    return final_features

# --- Class for caching extracted features from dataset ---
class PrecomputedFeatureDataset(Dataset):
    def __init__(self, features_tensor, values_tensor):
        self.features = features_tensor
        self.values = values_tensor
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.values[idx].unsqueeze(0)

# --- Multi-layer Perceptron Model ---
class EvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(376, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        ) 
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.net(x)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- Load precomputed features and values ---
    features_tensor, targets_tensor = torch.load("precomputed_features.pt")
    features = features_tensor.to(device)
    values = targets_tensor.to(device)

    data = PrecomputedFeatureDataset(features, values)

    total_size = len(data)
    train_size = int(0.8 * total_size)
    val_size = int(0.2 * total_size)
    train_dataset, val_dataset, _ = random_split(data, [train_size, val_size, total_size - train_size - val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=6)

    model = EvalNet().to(device)
    criterion = nn.MSELoss()

    num_of_epoch = 50
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait before stopping
    trigger_times = 0

    # Regularisation (might want change scheduler)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_epoch)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Store best model weights
    best_model_weights = None

    for epoch in range(num_of_epoch):
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
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # Save best model weights
            best_model_weights = model.state_dict().copy()
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break

        print(f"Epoch {epoch+1}/{num_of_epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(datetime.datetime.now())

        # # Sample printing logic
        # if (epoch + 1) % 5 == 0:
        #     all_data = load_data()
        #     sample_state, true_util = all_data[epoch]
        #     pred_util = model(torch_extract_features(sample_state).to(device).reshape(1, -1)).item()
        #     board_str = convert_board_to_string(sample_state.board)
        #     print("Sample State:")
        #     print(board_str)
        #     print(f"Predicted Utility: {pred_util:.4f} | True Utility: {true_util}")

    # Load best model weights before saving
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Save weights to JSON
    weights_dict = {}
    fc_count = 1
    bn_count = 1

    for layer in model.net:
        if isinstance(layer, nn.Linear):
            weights_dict[f"fc{fc_count}_weight"] = layer.weight.data.cpu().tolist()
            weights_dict[f"fc{fc_count}_bias"] = layer.bias.data.cpu().tolist()
            fc_count += 1
        elif isinstance(layer, nn.BatchNorm1d):
            weights_dict[f"bn{bn_count}_weight"] = layer.weight.data.cpu().tolist()
            weights_dict[f"bn{bn_count}_bias"] = layer.bias.data.cpu().tolist()
            weights_dict[f"bn{bn_count}_running_mean"] = layer.running_mean.data.cpu().tolist()
            weights_dict[f"bn{bn_count}_running_var"] = layer.running_var.data.cpu().tolist()
            bn_count += 1

    # Save weights dictionary
    with open('model_weights.json', 'w') as f:
        json.dump(weights_dict, f, indent=4)

    print("Model training complete and weights saved to 'model_weights.json'.")

if __name__ == "__main__":
    train_model()
