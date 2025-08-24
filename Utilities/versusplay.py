import time
import json
import numpy as np
from typing import Tuple, Optional
from utils import State, convert_board_to_string

# Define action type: (i, j, k, l)
Action = Tuple[int, int, int, int]

####################################################
# Fallback RandomAgent (if an agent times out on init)
####################################################
class RandomAgent:
    def __init__(self):
        self.evaluation_history = []
    def choose_action(self, state: State) -> Action:
        return state.get_random_valid_action()
    def evaluate(self, state: State) -> float:
        return 0.0

####################################################
# Helper for safe initialisation (max 1 second)
####################################################
def safe_init(agent_class, *args, **kwargs):
    start = time.time()
    try:
        agent = agent_class(*args, **kwargs)
    except Exception as e:
        print(f"Agent initialisation error: {e}")
        agent = None
    elapsed = time.time() - start
    if elapsed > 1.0 or agent is None:
        print("Agent initialisation exceeded 1 second. Using RandomAgent instead.")
        return RandomAgent()
    return agent

####################################################
# Game Match Function: One match between two agents
####################################################
def run_match(agent_first, agent_second):
    state = State()
    turn_count = 0
    evaluation_history_first = []
    evaluation_history_second = []
    
    while not state.is_terminal():
        turn_count += 1
        current_player = "Player 1" if state.fill_num == 1 else "Player 2"
        print(f"\n--- Turn {turn_count} ({current_player}) ---")
        print(convert_board_to_string(state.board))
        print("Meta Board Status:")
        print(state.local_board_status)
        
        # Determine which agent is playing this turn.
        if state.fill_num == 1:
            current_agent = agent_first
        else:
            current_agent = agent_second
        
        # --- Invert the state if necessary ---
        # Our agents always assume they are playing as player 1.
        # So if the current state's fill_num is not 1, invert it.
        if state.fill_num != 1:
            agent_state = state.invert()
        else:
            agent_state = state.clone()
        
        # Evaluate using the agent's perspective (always player 1)
        eval_val = current_agent.evaluate(agent_state)
        if state.fill_num == 1:
            evaluation_history_first.append(eval_val)
        else:
            evaluation_history_second.append(eval_val)
        print(f"Evaluation: {eval_val:.4f}")
        
        # Choose move from the agent's perspective.
        start_move = time.time()
        try:
            # The agent chooses an action based on agent_state,
            # which is inverted if necessary.
            action = current_agent.choose_action(agent_state)
        except Exception as e:
            print(f"Error in choose_action: {e}. Using random valid move.")
            action = state.get_random_valid_action()
        elapsed_move = time.time() - start_move
        if elapsed_move > 3.0:
            print(f"Move took {elapsed_move:.2f} sec. Using random valid move instead.")
            action = state.get_random_valid_action()
        if not state.is_valid_action(action):
            print("Chosen action is invalid. Using random valid move.")
            action = state.get_random_valid_action()
            
        # Apply the action to the true game state.
        state = state.change_state(action)
    
    print("\n=== Final Board ===")
    print(convert_board_to_string(state.board))
    print("Meta Board Status:")
    print(state.local_board_status)
    outcome = state.terminal_utility()
    return outcome, evaluation_history_first, evaluation_history_second, turn_count


####################################################
# Head-to-Head Runner: Two games with swapped roles
####################################################
def run_head_to_head(agent_class1, agent_class2):
    # Safely initialise both agents (each must finish __init__ within 1 second)
    agent1 = safe_init(agent_class1)
    agent2 = safe_init(agent_class2)
    
    print("=== Game 1: Agent1 as Player 1, Agent2 as Player 2 ===")
    outcome1, eval_hist1_game1, eval_hist2_game1, turns1 = run_match(agent1, agent2)
    
    print("\n=== Game 2: Agent2 as Player 1, Agent1 as Player 2 ===")
    outcome2, eval_hist2_game2, eval_hist1_game2, turns2 = run_match(agent2, agent1)
    
    print("\n=== Overall Results ===")
    def outcome_str(outcome, first_agent_name, second_agent_name):
        if outcome == 1.0:
            return f"{first_agent_name} wins"
        elif outcome == 0.0:
            return f"{second_agent_name} wins"
        else:
            return "Draw"
    
    print(f"Game 1 Outcome: {outcome_str(outcome1, 'Agent1', 'Agent2')}, Turns: {turns1}")
    print(f"Game 2 Outcome: {outcome_str(outcome2, 'Agent2', 'Agent1')}, Turns: {turns2}")
    
    print("\nAgent1 Combined Evaluation History:")
    combined_eval_agent1 = eval_hist1_game1 + eval_hist1_game2
    print([f"{val:.4f}" for val in combined_eval_agent1])
    
    print("\nAgent2 Combined Evaluation History:")
    combined_eval_agent2 = eval_hist2_game1 + eval_hist2_game2
    print([f"{val:.4f}" for val in combined_eval_agent2])

####################################################
# Main: Import your two agent classes and run head-to-head.
####################################################
if __name__ == "__main__":
    # Import agent classes from your files (adjust as needed)
    from agent2 import StudentAgent as Agent2
    from manualAgent import StudentAgent as Agent1
    
    run_head_to_head(Agent1, Agent2)
