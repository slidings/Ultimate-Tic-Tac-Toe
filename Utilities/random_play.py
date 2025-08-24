from manualAgent import StudentAgent
from utils import State, convert_board_to_string
import time
import numpy as np

class EvaluatingAgent(StudentAgent):
    def __init__(self, max_depth=3):
        super().__init__(max_depth)
        self.evaluation_history = []

    def choose_action(self, state: State) -> tuple[int, int, int, int]:
        current_eval = self.evaluate(state)
        self.evaluation_history.append((state.clone(), current_eval))
        return super().choose_action(state)

def run_vs_random(agent: EvaluatingAgent):
    print("Starting StudentAgent vs RandomBot game...\n")
    state = State()
    turn_count = 0
    
    while not state.is_terminal():
        turn_count += 1
        is_student_turn = (state.fill_num == 1)  # Assuming StudentAgent is always Player 1
        
        current_player = "StudentAgent" if is_student_turn else "RandomBot"
        print(f"\n=== Turn {turn_count} ({current_player}) ===")
        print(convert_board_to_string(state.board))
        print("\nMeta Board Status:")
        print(state.local_board_status)

        # Student evaluates current state
        if is_student_turn:
            eval_value = agent.evaluate(state)
            print(f"\nEvaluation: {eval_value:.4f}")
        print("-" * 50)

        # Get move
        if is_student_turn:
            start_time = time.time()
            action = agent.choose_action(state.clone())
            elapsed = time.time() - start_time
            if elapsed > agent.global_time_limit:
                print("Move timed out! Choosing random valid move.")
                action = state.get_random_valid_action()
        else:
            action = state.get_random_valid_action()
        
        if not state.is_valid_action(action):
            print("Invalid move! Choosing random move.")
            action = state.get_random_valid_action()
        
        # Apply move
        state = state.change_state(action)

    # Game ended
    print("\n=== Final Result ===")
    print(convert_board_to_string(state.board))
    print("\nMeta Board Status:")
    print(state.local_board_status)

    result = state.terminal_utility()
    if result == 1.0:
        print("\n🏆 StudentAgent (Player 1) wins!")
    elif result == 0.0:
        print("\n🤖 RandomBot (Player 2) wins!")
    else:
        print("\n🤝 It's a draw!")

    print(f"\nTotal turns: {turn_count}")
    
    # Show evaluations
    print("\n📈 StudentAgent Evaluation History:")
    for idx, (s, val) in enumerate(agent.evaluation_history):
        print(f"Turn {idx+1}: Eval = {val:.4f}")

if __name__ == "__main__":
    agent = EvaluatingAgent()
    run_vs_random(agent)
    
