import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from utils import load_data, convert_board_to_string
import json
import datetime
import numpy as np
import time
from utils import State, Action
from typing import Optional



WINNING_LINES = [
    # rows
    [(0,0), (0,1), (0,2)],
    [(1,0), (1,1), (1,2)],
    [(2,0), (2,1), (2,2)],
    # cols
    [(0,0), (1,0), (2,0)],
    [(0,1), (1,1), (2,1)],
    [(0,2), (1,2), (2,2)],
    # diag
    [(0,0), (1,1), (2,2)],
    [(0,2), (1,1), (2,0)]
]

def uncontested_lines(board):
    p1_uncontested = 0
    p2_uncontested = 0

    for line in WINNING_LINES:
        cells = [int(board[r, c]) for r, c in line]
        if 2 not in cells and 3 not in cells:
            p1_uncontested += 1
        if 1 not in cells and 3 not in cells:
            p2_uncontested += 1

    return p1_uncontested, p2_uncontested

# Player occupies at least 1 square in each winning line.
def potential_winning_lines(board):
    p1_count = 0
    p2_count = 0

    for line in WINNING_LINES:
        cells = [int(board[r, c]) for r, c in line]
        if 1 in cells and 2 not in cells and 3 not in cells:
            p1_count += 1
        if 2 in cells and 1 not in cells and 3 not in cells:
            p2_count += 1
    return p1_count, p2_count

# Player occupies at least 2 squares in each winning line.
def threats_features(board):
    p1_potential_wins = 0
    p2_potential_wins = 0
    
    # Convert board to numpy array for consistent handling
    board_np = np.array(board)
    
    for line in WINNING_LINES:
        # Extract cells using NumPy advanced indexing
        rows, cols = zip(*line)
        cells = board_np[rows, cols]
        
        # Count occurrences using vectorized operations
        count_p1 = np.sum(cells == 1)
        count_p2 = np.sum(cells == 2)
        count_empty = np.sum(cells == 0)
        
        # Check potential win conditions
        if count_p1 == 2 and count_empty == 1:
            p1_potential_wins += 1
        if count_p2 == 2 and count_empty == 1:
            p2_potential_wins += 1
    
    return p1_potential_wins, p2_potential_wins

def comprehensive_winning_lines_features(board,
                                         w_potential = 0.35,
                                         w_threats= 0.5,
                                         w_uncontested= 0.15):
    
    # Extract features
    p1_thr, p2_thr = threats_features(board)
    p1_pot, p2_pot = potential_winning_lines(board)
    p1_uncont, p2_uncont = uncontested_lines(board)

    # Per-feature normalized differences in [-1, 1]
    threat_score = (p1_thr - p2_thr) / 5.0
    potential_score = (p1_pot - p2_pot) / 8.0
    uncontested_score = (p1_uncont - p2_uncont) / 8.0

    # Weighted final score
    final_score = (
        w_threats * threat_score +
        w_potential * potential_score +
        w_uncontested * uncontested_score
    )
    return final_score


# def raw_fork_potential(local_board, player):
#     board_flat = local_board.flatten()
#     # Identify indices of empty cells
#     empty_idxs = np.where(board_flat == 0)[0]
#     fork_count = 0
#     for idx in empty_idxs:
#         candidate = board_flat.copy()
#         candidate[idx] = player
#         candidate_lines = candidate[WINNING_LINE_INDICES]
#         # Count the number of immediate wins (2 marks + 1 empty) in one go.
#         immediate_wins = np.sum((np.sum(candidate_lines == player, axis=1) == 2) & 
#                                 (np.sum(candidate_lines == 0, axis=1) == 1))
#         if immediate_wins >= 2:
#             fork_count += 1
#     return fork_count

# def enhanced_opponent_mobility(state):
#     """
#     For the forced local board (if available), returns a value such that:
#       - If the opponent can move freely (i.e. many empty cells and high winning potential),
#         the value is negative.
#       - If the opponent is restricted (i.e. few empty cells) and/or has little winning potential,
#         the value is positive.
    
#     This implementation computes the normalized free cell ratio on the forced local board
#     and scales it by the opponent's winning potential, then maps that to a range where full freedom
#     gives -1 and no freedom gives +1.
#     """
#     if state.prev_local_action is not None:
#         i, j = state.prev_local_action
#         if state.local_board_status[i][j] == 0:
#             local_board = state.board[i][j]
#             # Calculate the proportion of empty cells (free ratio)
#             free_ratio = np.sum(local_board == 0) / 9.0  # Value between 0 (no free cells) and 1 (completely empty)
            
#             # Compute opponent's winning potential on this board.
#             # Here we assume local_winning_lines_features returns counts of potential winning lines.
#             # For our purposes we use the opponent's count.
#             _, p2_win = local_winning_lines_features(local_board)
#             # Normalize by 8, since there are 8 possible winning lines in a 3x3 board.
#             mobility_factor = p2_win / 8.0  
            
#             # One simple mapping is to combine these factors so that when free_ratio and mobility_factor are high,
#             # the score becomes very negative (unfavorable), and when free_ratio is low, the score is high.
#             # For example, we can define:
#             score = 1.0 - 2.0 * free_ratio * mobility_factor
#             # Explanation:
#             #   - If free_ratio == 1 and mobility_factor == 1, then score = 1 - 2 = -1.
#             #   - If free_ratio == 0, then score = 1 (the opponent is not free to move).
#             #   - Intermediate values will map accordingly.
#             return score
#     # If there's no forced board, you might choose a neutral value (or adjust this default as needed).
#     return -0.5

# def urgency_weighted_near_completion(state):
#     total = 0.0
#     # Reshape to (9, 3, 3) to process all local boards.
#     boards = state.board.reshape(9, 3, 3)
#     for board in boards:
#         board_flat = board.flatten()
#         lines = board_flat[WINNING_LINE_INDICES]
#         # Compute for player 1 and player 2 in vectorized form.
#         p1_threats = np.sum((np.sum(lines == 1, axis=1) == 2) & (np.sum(lines == 0, axis=1) == 1))
#         p2_threats = np.sum((np.sum(lines == 2, axis=1) == 2) & (np.sum(lines == 0, axis=1) == 1))
#         total += (p1_threats * 1.5) - (p2_threats * 1.5)
#     return total / (9.0 * 2.0)

# ### DONE ###
# def dynamic_meta_board_winning_diff(state):
#     """
#     Computes the difference between available global winning lines for Player 1 and Player 2
#     on the meta-board, and normalizes by dividing by 8.
#     """
#     p1_global, p2_global = global_winning_lines_features(state.local_board_status)
#     return (p1_global - p2_global) / 8.0

# ### DONE ###
# def dynamic_meta_board_winning_threats(state):
#     p1_global, p2_global = global_threats_features(state.local_board_status)
#     return (p1_global - p2_global) / 5.0

# ### DONE unable to use, not implemented###
# def local_blocking_score(state):
#     if state.prev_action is None:
#         return 0.0
    
#     threat_diff = 0
#     potential_diff = 0
#     # Extract full coordinates from the 4-tuple.
#     meta_row, meta_col, local_row, local_col = state.prev_action

#     # Identify the current player (the one who made the move).
#     current_player = state.fill_num  
#     opponent = 3 - current_player

#     # Get the current (post-move) local board from the global board.
#     current_local_board = state.board[meta_row][meta_col]

#     # Reconstruct the pre-move local board by copying the current board and 'undoing' the move.
#     pre_local_board = current_local_board.copy()
#     pre_local_board[local_row, local_col] = 0

#     # This function returns (p1_threat, p2_threat).
#     pre_p1_threat, pre_p2_threat = local_threats_features(pre_local_board)
#     post_p1_threat, post_p2_threat = local_threats_features(current_local_board)

#     # This function returns winning lines
#     pre_p1_win_lines, pre_p2_win_lines = local_winning_lines_features(pre_local_board)
#     post_p1_win_lines, post_p2_win_lines = local_winning_lines_features(current_local_board)

#     if current_player == 2:
#         # Player 1 (us) made the previous move, we gain the threat_diff
#         # Want to maximise threat_diff (keep it positive)
#         threat_diff = pre_p2_threat - post_p2_threat
#         potential_diff = pre_p2_win_lines - post_p2_win_lines

#     else:
#         # Opponent made previous move, if its good we return more negative
#         threat_diff = post_p1_threat - pre_p1_threat
#         potential_diff = post_p1_win_lines - pre_p1_win_lines
    
#     return (threat_diff * 0.8 + potential_diff * 0.2) / 5.0

# ### DONE, unable to use, not implemented ###
# def global_blocking_score(state):
#     if state.prev_action is None:
#         return 0.0
    
#     threat_diff = 0
#     potential_diff = 0

#     meta_row, meta_col, local_row, local_col = state.prev_action

#     current_player = state.fill_num  
#     opponent = 3 - current_player

#     # Get the current (post-move) meta board from the global board.
#     current_meta_board = state.local_board_status  # This is the status of all local boards.
    
#     # Reconstruct the pre-move meta board by copying the current status and 'undoing' the move.
#     pre_meta_board = current_meta_board.copy()
    
#     # Undo the action on the meta board
#     pre_meta_board[meta_row][meta_col] = 0  # Unset the meta-board status. (Definitely not won)

#     # This function returns (p1_threat, p2_threat).
#     pre_p1_threat, pre_p2_threat = local_threats_features(pre_meta_board)
#     post_p1_threat, post_p2_threat = local_threats_features(current_meta_board)

#     # This function returns winning lines
#     pre_p1_win_lines, pre_p2_win_lines = local_winning_lines_features(pre_meta_board)
#     post_p1_win_lines, post_p2_win_lines = local_winning_lines_features(current_meta_board)

#     # Now calculate the threat and potential differences
#     if current_player == 2:
#         # Player 1 (us) made the previous move, we gain the threat_diff
#         threat_diff = pre_p2_threat - post_p2_threat
#         potential_diff = pre_p2_win_lines - post_p2_win_lines
#     else:
#         # Opponent made previous move, if it's good we return more negative
#         threat_diff = post_p1_threat - pre_p1_threat
#         potential_diff = post_p1_win_lines - pre_p1_win_lines
    
#     return (threat_diff * 0.8 + potential_diff * 0.2) / 8.0

# ## Just changed from binary to scale ##
# def force_opponent_into_trap(state):
#     """
#     Returns normalized difference between player 1 and player 2 trap potential.
#     """
#     if state.prev_local_action is None:
#         return 0.0
    
#     i, j = state.prev_local_action
#     if state.local_board_status[i][j] != 0:
#         return 0.0

#     local_board = state.board[i][j]
#     p1_traps, p2_traps = local_threats_features(local_board)
#     return (p1_traps - p2_traps) / 8.0

# ### DONE ###
# def force_opponent_to_fork_board(state):
#     """
#     Returns 1.0 if the opponent is forced to a board where we (player 1) have fork potential.
#     """
#     if state.prev_local_action is None:
#         # Impossible to have forks in the starting move
#         return 0.0
    
#     i, j = state.prev_local_action
#     if state.local_board_status[i][j] != 0:
#         return 0.0

#     local_board = state.board[i][j]
#     fork_score = raw_fork_potential(local_board, player=1)

#     # not sure if returning fork_score directly will be better?
#     # does it matter if someone has many forks?
#     return fork_score

# ### DONE ###
# def center_board_control(state):
#     """
#     Returns a feature for meta center board control.
#     Evaluates if the center local board is won, near completion, or contested.
#     """
#     center_board = state.board[1][1]

#     # Importance of near win vs winning potential, to be tuned.
#     # Ensure that when its empty it should be close to draw.
#     #
#     # But also the winning potential between 2 should be considered even if 
#     # there isn't 2 in a row yet

#     #TODO
#     hyperparameter = 0.8

#     score = 0.0
#     if state.local_board_status[1][1] == 0:
#         # Ongoing
#         # Go for winning lines + near wins

#         p1_threats, p2_threats = local_threats_features(center_board)
#         threats_score = (p1_threats - p2_threats) / 7.0

#         p1_win, p2_win = local_winning_lines_features(center_board)
#         winning_potential_score = (p1_win - p2_win) / 8.0

#         score = threats_score * hyperparameter + winning_potential_score * (1 - hyperparameter)

#     elif state.local_board_status[1][1] == 1:
#         # win for p1
#         score = 1.0

#     elif state.local_board_status[1][1] == 2:
#         # win for p2
#         score = -1.0

#     else:
#         # draw
#         score = 0.0
    
#     return score

# ### DONE ###
# def corner_board_control(state):
#     """
#     Returns a feature for meta corner boards control.
#     Evaluates if the corner local boards are won, near completion, or contested.
#     Uses a loop to calculate the score for each corner board and returns the average score.
#     """
#     # List of coordinates for the four corner boards (meta board)
#     corner_coords = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
#     # Hyperparameter for weighting between threats and winning potential
#     hyperparameter = 0.8

#     total_score = 0.0
#     corner_count = 0

#     # Loop through each corner position and compute the score
#     for (i, j) in corner_coords:
#         corner_board = state.board[i][j]
#         status = state.local_board_status[i][j]  # status of the corner board
        
#         if status == 0:  # Ongoing game in this corner
#             p1_threats, p2_threats = local_threats_features(corner_board)
#             threats_score = (p1_threats - p2_threats) / 7.0

#             p1_win, p2_win = local_winning_lines_features(corner_board)
#             winning_potential_score = (p1_win - p2_win) / 8.0  # Normalize by max number of winning lines in 3x3 grid

#             # Weighted score between threats and winning potential
#             score = threats_score * hyperparameter + winning_potential_score * (1 - hyperparameter)

#         elif status == 1:  # Player 1 wins this corner
#             score = 1.0

#         elif status == 2:  # Player 2 wins this corner
#             score = -1.0

#         else:  # Draw (status == 3)
#             score = 0.0
        
#         total_score += score
#         corner_count += 1

#     # Return the average score for all corner boards
#     return total_score / corner_count

# #### DONE ####
# def free_move_advantage(state):
#     """
#     Returns a feature indicating free move advantage. If state.prev_local_action is None,
#     the player has a free move; otherwise 0.
#     """
#     prev_move = state.prev_local_action
#     if prev_move is None:
#         # First move
#         return 1.0
#     else:
#         i, j = prev_move
#         # Game has concluded
#         if state.local_board_status[i][j] != 0:
#             return 1.0
#         else:
#             # Still on going
#             return 0.0

# # Feature A
# def global_winning_line_diff(meta_board) -> float:
#     p1, p2 = global_winning_lines_features(meta_board)
    
#     # Normalise by max winning lines
#     return (p1 - p2) / 8.0

# # Feature B
# def global_threats_diff(global_board) -> float:
#     p1_threat, p2_threat = global_threats_features(global_board)
#     return (p1_threat - p2_threat) / 8.0

# # Feature C
# def meta_board_occupation(meta_board) -> float:
#     """
#     meta_board: shape (3,3) tensor with values 0 (ongoing), 1 (won by player1), 2 (won by player2).
#     Returns normalized difference: (p1_won - p2_won) / 9.
#     """
#     p1_won = np.sum(meta_board == 1)
#     p2_won = np.sum(meta_board == 2)
#     return (p1_won - p2_won) / 9.0

# # Feature D
# def meta_board_forks(meta_board) -> float:
#     # Max number of forks = 5
#     return ((raw_fork_potential(meta_board, 1) - raw_fork_potential(meta_board, 2))) / 5.0
    
# # Feature E
# def opponents_mobility(state) -> float:
#     return enhanced_opponent_mobility(state)

# # Feature F
# # Blocking
# def blocking_score_by_player(state, player: int) -> float:
#     """
#     Computes the blocking score for the given player on the meta board.
    
#     For a 3x3 meta board (state.local_board_status, a NumPy array),
#     a winning line is considered "blocked" by the player if:
#         - At least one cell in the line equals the player's piece, AND
#         - None of the cells in that line belong to the opponent (which is 3 - player).
    
#     The function returns the fraction of winning lines (out of 8) that are blocked.
    
#     Args:
#         state: The game state with attribute `local_board_status` (NumPy array).
#         player (int): The player's marker (e.g. 1 or 2).
        
#     Returns:
#         float: Normalized blocking score.
#     """
#      # Get the forced local board location
#     if state.prev_local_action is None:
#         return 0.0
    
#     target_i, target_j = state.prev_local_action
    
#     # Check if board is still playable
#     if state.local_board_status[target_i][target_j] != 0:
#         return 0.0
    
#     local_board = state.board[target_i][target_j]
#     opponent = 3 - player
    
#     # Calculate current opponent threats
#     _, current_threats = local_threats_features(local_board) if player == 1 \
#         else local_threats_features(local_board)[::-1]
    
#     if current_threats == 0:
#         return 0.0
    
#     max_reduction = 0
    
#     # Try all possible moves in the local board
#     for x in range(3):
#         for y in range(3):
#             if local_board[x][y] != 0:
#                 continue  # Skip occupied cells
                
#             # Simulate placing player's piece
#             simulated_board = local_board.copy()
#             simulated_board[x][y] = player
            
#             # Calculate new threats
#             new_p1_threats, new_p2_threats = local_threats_features(simulated_board)
#             new_threats = new_p2_threats if player == 1 else new_p1_threats
            
#             threat_reduction = current_threats - new_threats
#             max_reduction = max(max_reduction, threat_reduction)

#     # Normalize by maximum possible reduction
#     return max_reduction / current_threats

# def blocking_score_difference(state) -> float:
#     """
#     Computes the absolute difference between your blocking score and your opponent's blocking score,
#     where each score is normalized to fall between 0 and 1.
    
#     Your blocking score is computed by calculating the fraction of winning lines (on the 3x3 meta board)
#     that are blocked by your pieces, and similarly for the opponent.
    
#     Returns:
#         float: A value between 0 and 1 representing the absolute difference between the two blocking scores.
#     """
#     my_piece = state.fill_num
#     opp_piece = 3 - state.fill_num
    
#     my_score = blocking_score_by_player(state, my_piece)
#     opp_score = blocking_score_by_player(state, opp_piece)

#     return my_score - opp_score

# # Feature G: send penalty
# def send_penalty(state):

#     # Currently opponent's turn
#     if state.prev_local_action is None:
#         # Starting move will never have dead board
#         return 0.0
    
#     i, j = state.prev_local_action

#     if state.local_board_status[i][j] != 0:
#         # The board has ended
#         if (state.local_board_status[i][j] == (3 - state.fill_num)):
#             # Won, not 1.0 since opponent still gets free move
#             return 0.5
#         if (state.local_board_status[i][j] == 3):
#             # Drawn board, should be avoided, since no benefit of drawing
#             return -0.25
#         # There is no way it can be opponent's won board (non valid move)
        
#     local_board = state.board[i][j]
#     p1_lines, p2_lines = local_winning_lines_features(local_board)
#     p1_threats, p2_threats = local_threats_features(local_board)

#     return (p1_lines - p2_lines) * 0.2 + (p1_threats - p2_threats) * 0.8

# # Feature H:
# def center_board_control(state):
#     """
#     Returns a feature for meta center board control.
#     Evaluates if the center local board is won, near completion, or contested.
#     """
#     center_board = state.board[1][1]

#     # Importance of near win vs winning potential, to be tuned.
#     # Ensure that when its empty it should be close to draw.
#     #
#     # But also the winning potential between 2 should be considered even if 
#     # there isn't 2 in a row yet

#     #TODO
#     hyperparameter = 0.8

#     score = 0.0
#     if state.local_board_status[1][1] == 0:
#         # Ongoing
#         # Go for winning lines + near wins

#         p1_threats, p2_threats = local_threats_features(center_board)
#         threats_score = (p1_threats - p2_threats) / 5.0

#         p1_win, p2_win = local_winning_lines_features(center_board)
#         winning_potential_score = (p1_win - p2_win) / 8.0

#         score = threats_score * hyperparameter + winning_potential_score * (1 - hyperparameter)

#     elif state.local_board_status[1][1] == 1:
#         # win for p1
#         score = 1.0

#     elif state.local_board_status[1][1] == 2:
#         # win for p2
#         score = -1.0

#     else:
#         # draw
#         score = 0.0
    
#     return score

# # Feature I
# def corner_board_control(state):
#     """
#     Returns a feature for meta corner boards control.
#     Evaluates if the corner local boards are won, near completion, or contested.
#     Uses a loop to calculate the score for each corner board and returns the average score.
#     """
#     # List of coordinates for the four corner boards (meta board)
#     corner_coords = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
#     # Hyperparameter for weighting between threats and winning potential
#     hyperparameter = 0.8

#     total_score = 0.0
#     corner_count = 0

#     # Loop through each corner position and compute the score
#     for (i, j) in corner_coords:
#         corner_board = state.board[i][j]
#         status = state.local_board_status[i][j]  # status of the corner board
        
#         if status == 0:  # Ongoing game in this corner
#             p1_threats, p2_threats = local_threats_features(corner_board)
#             threats_score = (p1_threats - p2_threats) / 5.0  # Normalize by max number of threats in 3x3 grid

#             p1_win, p2_win = local_winning_lines_features(corner_board)
#             winning_potential_score = (p1_win - p2_win) / 8.0  # Normalize by max number of winning lines in 3x3 grid

#             # Weighted score between threats and winning potential
#             score = threats_score * hyperparameter + winning_potential_score * (1 - hyperparameter)

#         elif status == 1:  # Player 1 wins this corner
#             score = 1.0

#         elif status == 2:  # Player 2 wins this corner
#             score = -1.0

#         else:  # Draw (status == 3)
#             score = 0.0
        
#         total_score += score
#         corner_count += 1

#     # Return the average score for all corner boards
#     return total_score / corner_count
# # Feature J
# def weighted_piece_diff(board: np.ndarray) -> float:
#     """
#     Compute the normalized weighted difference for a local board using NumPy.
#     Hard-coded weights:
#        Corners: 3, Edges: 2, Center: 4.
#     Normalized by total (24) so that the output is in [-1, 1].

#     Args:
#         board (np.ndarray): 3x3 array representing the board (values: 0, 1, or 2)

#     Returns:
#         float: Weighted difference in [-1, 1] range
#     """
#     weights = np.array([[3.0, 2.0, 3.0],
#                         [2.0, 4.0, 2.0],
#                         [3.0, 2.0, 3.0]])
#     total_weight = weights.sum()  # 24.0

#     p1_weight = np.sum((board == 1) * weights)
#     p2_weight = np.sum((board == 2) * weights)

#     diff = (p1_weight - p2_weight) / total_weight
#     return diff

# def local_boards_score(state):
#     # Hyper parameter to tune 
#     a = 0.8
#     p1_total_threats = 0
#     p2_total_threats = 0
#     p1_total_lines = 0
#     p2_total_lines = 0
#     meta_board = state.local_board_status

#     # number of better boards
#     p1_boards = 0
#     p2_boards = 0

#     for i in range(3):
#         for j in range(3):
#             if meta_board[i][j] != 0:
#                 # Finalised (Counted in meta_board_occupation)
#                 continue
#             local_board = state.board[i][j]
#             p1_threats, p2_threats = local_threats_features(local_board)
#             p1_lines, p2_lines = local_winning_lines_features(local_board)

#             if (p1_threats - p2_threats) * 0.8 + (p1_lines - p2_lines) * 0.2 > 0:
#                 p1_boards += 1
#             else:   
#                 p2_boards += 1

#     #         p1_total_lines += p1_lines
#     #         p2_total_lines += p2_lines
#     #         p1_total_threats += p1_threats
#     #         p2_total_threats += p2_threats
    
#     # final_score = (p1_total_threats - p2_total_threats) * a + (p1_total_lines - p2_total_lines) * (1 - a)
#     # # maximum 8 lines * 9 boards
#     # return final_score / 72.0
#     return (p1_boards - p2_boards) / 9.0

# def turn_number(state):
#     return np.sum(state.board == 1)


# def mobility_ratio(state: State) -> float:
#     """Calculates move advantage between players"""
#     p1_actions = len(state.get_all_valid_actions())
#     p2_state = state.invert()
#     p2_actions = len(p2_state.get_all_valid_actions())
#     return (p1_actions - p2_actions) / (p1_actions + p2_actions + 1e-6)

# --- Student Agent with Alpha-Beta Search ---
class StudentAgent:
    def __init__(self, max_depth=8):
        self.global_time_limit = 3.0
        self.max_depth = max_depth
        self.time_limit = 10
        self.best_action = None
        self.best_value = -np.inf
        self.first_move_done = False

        self.meta_patterns = [
            # Rows
            [(0,0), (0,1), (0,2)],
            [(1,0), (1,1), (1,2)],
            [(2,0), (2,1), (2,2)],
            # Columns
            [(0,0), (1,0), (2,0)],
            [(0,1), (1,1), (2,1)],
            [(0,2), (1,2), (2,2)],
            # Diagonals
            [(0,0), (1,1), (2,2)],
            [(0,2), (1,1), (2,0)]
        ]
        self.corners = [(0,0), (0,2), (2,0), (2,2)]

        # Weights for different features
        self.WEIGHTS = {
            'meta_near_win': 0.3,             # One move away from winning meta-board
            'meta_double_threat': 0.2,        # Two ways to win the meta-board
            'local_board_wins': 0.15,         # Value of winning local boards
            'local_near_win': 0.05,           # Value of being one move from winning local board
            'center_board_control': 0.1,      # Value of controlling center board
            'corner_boards_control': 0.05,    # Value of controlling corner boards
            'free_move_advantage': 0.075,     # Value of having a free move choice
            'next_move_score': 0.075          # Value of the position after next move
        }

    def choose_action(self, state: State) -> Action:
        self.start_time = time.time()
        self.best_action = None
        self.best_value = -np.inf
        valid_actions = state.get_all_valid_actions()
        
        if not valid_actions:
            return state.get_random_valid_action()
        
        if not self.first_move_done and np.all(state.board == 0):
            self.first_move_done = True
            return (1, 1, 1, 1)
        
        ordered_moves = []
        for action in valid_actions:
            if time.time() - self.start_time > self.time_limit:
                break
            next_state = state.clone().change_state(action)
            if action not in state.get_all_valid_actions():
                continue
            h_val = self.evaluate(next_state)
            ordered_moves.append((action, h_val))
            if h_val > self.best_value:
                self.best_value = h_val
                self.best_action = action
                
        ordered_moves.sort(key=lambda x: x[1], reverse=True)
        sorted_actions = [a for a, _ in ordered_moves]
        if not sorted_actions:
            sorted_actions = valid_actions
        
        try:
            current_depth = 1
            while current_depth <= self.max_depth and time.time() - self.start_time < self.time_limit:
                alpha = -np.inf
                beta = np.inf
                depth_best_value = -np.inf
                depth_best_action = None
                for action in sorted_actions:
                    if time.time() - self.start_time > self.time_limit:
                        break
                    next_state = state.clone().change_state(action)
                    if action not in state.get_all_valid_actions():
                        continue
                    val, _ = self.minimax(next_state, current_depth, alpha, beta, False)
                    if val > depth_best_value:
                        depth_best_value = val
                        depth_best_action = action
                        alpha = max(alpha, val)
                    if val > self.best_value:
                        self.best_value = val
                        self.best_action = action
                print(f"Depth {current_depth} best value: {depth_best_value}")
                current_depth += 1
        except TimeoutError:
            pass
        
        if self.best_action not in state.get_all_valid_actions():
            return state.get_random_valid_action()
        return self.best_action or sorted_actions[0]

    def minimax(self, state: State, depth: int, alpha: float, beta: float, maximizing: bool) -> tuple[float, Optional[Action]]:
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError("Search time limit exceeded in minimax")
        if depth == 0 or state.is_terminal():
            return self.evaluate(state), None
        
        valid_actions = state.get_all_valid_actions()
        best_val = -np.inf if maximizing else np.inf
        best_action = None

        for action in valid_actions:
            if time.time() - self.start_time > self.time_limit:
                break
            next_state = state.clone().change_state(action)
            val, _ = self.minimax(next_state, depth - 1, alpha, beta, not maximizing)
            if maximizing:
                if val > best_val:
                    best_val = val
                    best_action = action
                    alpha = max(alpha, best_val)
            else:
                if val < best_val:
                    best_val = val
                    best_action = action
                    beta = min(beta, best_val)
            if alpha >= beta:
                break
        return best_val, best_action

    def evaluate(self, state: State) -> float:
        """
        Evaluates the current state of an Ultimate Tic-Tac-Toe game.
        
        Parameters:
        - state: The current State object of the Ultimate TTT game
        
        Returns:
        - score: Float value between 0 and 1 where higher values favor player 1
                and lower values favor player 2
        """
        # If game is terminal, return the actual utility
        if state.is_terminal():
            return state.terminal_utility()
        
        
        board = state.board
        local_board_status = state.local_board_status
        fill_num = state.fill_num
        prev_local_action = state.prev_local_action
        
        # Neutral score
        score = 0
        
        # --- 1. Evaluate meta board near-wins and potential lines ---
        p1_near_wins = 0
        p2_near_wins = 0

        p1_potential_lines = 0
        p2_potential_lines = 0

        for pattern in self.meta_patterns:
            p1_count = sum(1 for r, c in pattern if local_board_status[r][c] == 1)
            p2_count = sum(1 for r, c in pattern if local_board_status[r][c] == 2)
            empty_count = sum(1 for r, c in pattern if local_board_status[r][c] == 0)
            
            # 1 move away
            if p1_count == 2 and empty_count == 1:
                p1_near_wins += 1
            elif p2_count == 2 and empty_count == 1:
                p2_near_wins += 1

            # 2 move away
            if p1_count == 1 and empty_count == 2:
                p1_potential_lines += 1
            elif p2_count == 1 and empty_count == 2:
                p2_potential_lines += 1
        
        comprehensive_score =  ((p1_near_wins - p2_near_wins) / 5.0) * 0.7 + ((p1_potential_lines - p2_potential_lines) / 8.0) * 0.3
        
        # Adjust score based on comprehensive scores
        score += self.WEIGHTS['meta_near_win'] * comprehensive_score
        
        # 2. Fork score
        if p1_near_wins > 1:
            score += self.WEIGHTS['meta_double_threat'] * (p1_near_wins - 1) / 5.0
        if p2_near_wins > 1:
            score -= self.WEIGHTS['meta_double_threat'] * (p2_near_wins - 1) / 5.0
        
        # 3. Number of won boards on meta board
        p1_wins = np.sum(local_board_status == 1)
        p2_wins = np.sum(local_board_status == 2)
        score += self.WEIGHTS['local_board_wins'] * (p1_wins - p2_wins) / 9
        
        # --- 3. Evaluate near-wins on local boards ---
        p1_local_near_wins = 0
        p2_local_near_wins = 0
        
        for i in range(3):
            for j in range(3):
                # Skip boards that are already won
                if local_board_status[i][j] != 0:
                    continue
                
                local_board = board[i][j]
                
                # Check each pattern on the local board
                for pattern in self.meta_patterns:  # Reuse the same patterns
                    positions = [(r, c) for r, c in pattern]
                    
                    # Count pieces in the pattern
                    p1_count = sum(1 for r, c in positions if local_board[r][c] == 1)
                    p2_count = sum(1 for r, c in positions if local_board[r][c] == 2)
                    empty_count = sum(1 for r, c in positions if local_board[r][c] == 0)
                    
                    # Near-win patterns
                    if p1_count == 2 and empty_count == 1:
                        p1_local_near_wins += 1
                    elif p2_count == 2 and empty_count == 1:
                        p2_local_near_wins += 1
        
        score += self.WEIGHTS['local_near_win'] * (p1_local_near_wins - p2_local_near_wins) / 72  # Max 8 patterns * 9 boards
        
        # --- 4. Center board control ---
        center_value = 0
        if local_board_status[1][1] == 1:
            center_value = 1
        elif local_board_status[1][1] == 2:
            center_value = -1
        elif local_board_status[1][1] == 0:
            # If center board not won, evaluate near-wins on it
            local_board = board[1][1]
            p1_center_near_wins = 0
            p2_center_near_wins = 0
            
            for pattern in self.meta_patterns:
                positions = [(r, c) for r, c in pattern]
                p1_count = sum(1 for r, c in positions if local_board[r][c] == 1)
                p2_count = sum(1 for r, c in positions if local_board[r][c] == 2)
                empty_count = sum(1 for r, c in positions if local_board[r][c] == 0)
                
                if p1_count == 2 and empty_count == 1:
                    p1_center_near_wins += 1
                elif p2_count == 2 and empty_count == 1:
                    p2_center_near_wins += 1
            
            center_value = (p1_center_near_wins - p2_center_near_wins) / 8
        
        score += self.WEIGHTS['center_board_control'] * center_value
        
        # --- 5. Corner boards control ---
        corner_value = 0
        
        for r, c in self.corners:
            if local_board_status[r][c] == 1:
                corner_value += 1
            elif local_board_status[r][c] == 2:
                corner_value -= 1
        
        score += self.WEIGHTS['corner_boards_control'] * corner_value / 4
        
        # --- 6. Free move advantage ---
        free_move_score = 0
        if prev_local_action is None:
            # Initial state - no advantage
            pass
        else:
            prev_row, prev_col = prev_local_action
            # If next board is already won or tied (3), player has free move
            if local_board_status[prev_row][prev_col] != 0:
                # Current player's advantage
                if fill_num == 1:
                    free_move_score = 1
                else:
                    free_move_score = -1
            else:
                # Check if being sent to a board with near-wins for opponent
                local_board = board[prev_row][prev_col]
                opponent_near_wins = 0
                
                for pattern in self.meta_patterns:
                    positions = [(r, c) for r, c in pattern]
                    opp_count = sum(1 for r, c in positions if local_board[r][c] == (3 - fill_num))
                    empty_count = sum(1 for r, c in positions if local_board[r][c] == 0)
                    
                    if opp_count == 2 and empty_count == 1:
                        opponent_near_wins += 1
                
                # Negative score if many opponent near-wins
                if opponent_near_wins > 0:
                    free_move_score = -opponent_near_wins / 8
                    if fill_num == 2:  # Flip the sign for player 2
                        free_move_score = -free_move_score
        
        score += self.WEIGHTS['free_move_advantage'] * free_move_score
        
        return score