import numpy as np
from dataclasses import dataclass
import pickle

# First 2 represents which sub-board, last 2 represents pos within sub-board
Action = tuple[int, int, int, int]

# Specifies move's location within sub-board
LocalBoardAction = tuple[int, int]


def convert_board_to_string(board):
    output = []
    horizontal_separator = "-" * 21
    
    for super_row in range(3):
        for sub_row in range(3):
            line_parts = []
            for super_col in range(3):
                sub_board = board[super_row][super_col]
                sub_line = " ".join(str(sub_board[sub_row][sub_col]) for sub_col in range(3))
                line_parts.append(sub_line)
            full_line = " | ".join(line_parts)
            output.append(full_line)
        if super_row != 2:
            output.append(horizontal_separator)
    
    return '\n'.join(output)

ENDLINE = '\n'

# Immutable object, attributes cannot be changed
@dataclass(frozen=True)
class ImmutableState:
    # Attributes of this class
    board: np.ndarray
    prev_local_action: LocalBoardAction | None
    fill_num: 1 | 2
    # calculated after init to indicate status for each subboard
    local_board_status: np.ndarray = None

    ### METHODS ###

    # calculates local_board_status after object created
    def __post_init__(self):
        object.__setattr__(self, 'local_board_status', get_local_board_status(self.board))

    # Checks if 2 boards are the same
    def __eq__(self, other):
        return np.all(self.board == other.board) and self.prev_local_action == other.prev_local_action and self.fill_num == other.fill_num

    # Formatted string representation for debugging
    def __repr__(self):
        return f"""State(
    board=
        {convert_board_to_string(self.board).replace(ENDLINE, ENDLINE+'        ')}, 
    local_board_status=
        {str(self.local_board_status).replace(ENDLINE, ENDLINE+'        ')}, 
    prev_local_action={self.prev_local_action}, 
    fill_num={self.fill_num}
)
"""

# Given an action, extracts and returns the local pos as 2 int tuple
def get_local_board_action(action: Action) -> LocalBoardAction:
    meta_row, meta_col, local_row, local_col = action
    return LocalBoardAction((local_row, local_col))

# [Support Method] checks specific local board if there is 3 in a row and output the result of the board
def board_status(board: np.ndarray) -> int:
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return 0
    return 3

# Applies board_status to every subboard, produces a new array indicating status of each SB
def get_local_board_status(board: np.ndarray) -> None:
    local_board_status: np.ndarray = np.array([[0 for i in range(3)] for j in range(3)])
    for i in range(3):
        for j in range(3):
            local_board_status[i][j] = board_status(board[i][j])
    return local_board_status

# Checks action validity based on data type, len, correct subboard, empty cell
def is_valid_action(state: ImmutableState, action: Action) -> bool:
    if not isinstance(action, tuple):
        return False
    if len(action) != 4:
        return False
    i, j, k, l = action
    if type(i) != int or type(j) != int or type(k) != int or type(l) != int:
        return False
    # Means the game has concluded
    if state.local_board_status[i][j] != 0:
        return False
    if state.board[i][j][k][l] != 0:
        return False
    if state.prev_local_action is None:
        return True
    prev_row, prev_col = state.prev_local_action
    if prev_row == i and prev_col == j:
        return True
    return state.local_board_status[prev_row][prev_col] != 0

# [Support Method] Goes through every cell on every subboard that isn't finished
def _get_all_valid_free_actions(state: ImmutableState) -> list[Action]:
    valid_actions: list[Action] = []
    for i in range(3):
        for j in range(3):
            if state.local_board_status[i][j] != 0:
                continue
            for k in range(3):
                for l in range(3):
                    if state.board[i][j][k][l] == 0:
                        valid_actions.append((i, j, k, l))
    return valid_actions

# Covers first moves and subsequent moves
def get_all_valid_actions(state: ImmutableState) -> list[Action]:
    # If no previous action or target local board is finished
    if state.prev_local_action is None:
        return _get_all_valid_free_actions(state)
    prev_row, prev_col = state.prev_local_action
    if state.local_board_status[prev_row][prev_col] != 0:
        return _get_all_valid_free_actions(state)
    
    # Choose based on previously defined move
    valid_actions: list[Action] = []
    for i in range(3):
        for j in range(3):
            if state.board[prev_row][prev_col][i][j] == 0:
                valid_actions.append((prev_row, prev_col, i, j))
    return valid_actions

# Flips between 1 and 2, next player's turn
def get_next_turn_fill_num(fill_num):
    return 3 - fill_num

# Random valid moves from the list
def get_random_valid_action(state: ImmutableState) -> Action:
    valid_actions = get_all_valid_actions(state)
    return valid_actions[np.random.randint(len(valid_actions))]

# Checks validity of move, updates the board with a new immutablestate
def change_state(state: ImmutableState, action: Action, check_valid_action = True) -> ImmutableState:
    if check_valid_action:
        assert is_valid_action(state, action), f"Invalid action: {action}"
    i, j, k, l = action
    new_board = state.board.copy()
    new_board[i][j][k][l] = state.fill_num
    new_state = ImmutableState(board=new_board, fill_num=get_next_turn_fill_num(state.fill_num), prev_local_action=get_local_board_action(action))
    return new_state

# If the game has ended based on meta board
def is_terminal(state: ImmutableState) -> bool:
    return board_status(state.local_board_status) != 0

# Utility wrt agent
def terminal_utility(state: ImmutableState) -> float:
    status = board_status(state.local_board_status)
    if status == 1:
        return 1.0
    if status == 2:
        return 0.0
    if status == 3:
        return 0.5
    assert False, "Board is not terminal"

# Swap 1 to 2
def invert(state: ImmutableState) -> ImmutableState:
    board = np.zeros_like(state.board)
    board[state.board == 1] = 2
    board[state.board == 2] = 1
    return ImmutableState(
        board = board,
        prev_local_action = state.prev_local_action,
        fill_num = get_next_turn_fill_num(state.fill_num)
    )

# Higher level class that warps around immutable state
class State: # Wrapper for ImmutableState
    def __init__(self,
                 board: np.ndarray = None,
                 fill_num: 1 | 2 = 1,
                 prev_action: Action | None = None,
                 prev_local_action: LocalBoardAction | None = None,
                 local_board_status: np.ndarray = None, # deprecated: only for backward compatibility
                 ):
        if board is None:
            board = np.array([[[[0 for i in range(3)]for j in range(3)] for k in range(3)] for l in range(3)])
        if prev_local_action is None and prev_action is not None:
            prev_local_action = get_local_board_action(prev_action)
        if local_board_status is not None:
            print("Warning: The use of local_board_status is deprecated and will be ignored.")
        self._state = ImmutableState(board=board, fill_num=fill_num, prev_local_action=prev_local_action)

    def __eq__(self, other):
        return self._state == other._state

    def __repr__(self):
        return self._state.__repr__()

    @property
    def board(self) -> np.array:
        return self._state.board

    @property
    def fill_num(self) -> 1 | 2:
        return self._state.fill_num

    @property
    def local_board_status(self) -> np.array:
        return self._state.local_board_status

    @property
    def prev_local_action(self) -> np.array:
        return self._state.prev_local_action

    def update_local_board_status(self) -> None:
        pass # Does nothing, only for backward compatibility

    def get_backward_compatible_state(self, prev_action: Action | None = None) -> ImmutableState:
        state = self._state
        if prev_action is not None and get_local_board_action(prev_action) != self._state.prev_local_action:
            print("Warning: The prev_action you specified contains a local action that differs from the one in the current state.")
            state = ImmutableState(board=self._state.board, fill_num=self._state.fill_num,
                                prev_local_action=get_local_board_action(prev_action))
        return state

    def is_valid_action(self, action: Action, prev_action: Action | None = None) -> bool:
        return is_valid_action(self.get_backward_compatible_state(prev_action), action)

    def get_all_valid_actions(self, prev_action: Action | None = None) -> list[Action]:
        return get_all_valid_actions(self.get_backward_compatible_state(prev_action))

    def _get_all_valid_free_actions(self) -> list[Action]:
        return _get_all_valid_free_actions(self._state)

    def get_random_valid_action(self, prev_action: Action | None = None) -> Action:
        return get_random_valid_action(self.get_backward_compatible_state(prev_action))

    def change_state(self, action: Action, prev_action: Action | None = None, in_place: bool = False, check_valid_action = True) -> "State":
        if in_place: # break backward compatibility
            raise NotImplementedError
        new_state = change_state(self.get_backward_compatible_state(prev_action), action)
        return State(board=new_state.board, fill_num=new_state.fill_num, prev_local_action=new_state.prev_local_action)

    def is_terminal(self) -> bool:
        return is_terminal(self._state)

    def terminal_utility(self) -> float:
        return terminal_utility(self._state)

    def invert(self) -> "State":
        new_state = invert(self._state)
        return State(board=new_state.board, fill_num=new_state.fill_num, prev_local_action=new_state.prev_local_action)

    def clone(self) -> "State":
        return State(
            board=self._state.board.copy(),
            fill_num=self._state.fill_num,
            prev_local_action=self._state.prev_local_action
        )


def load_data() -> list[tuple[State, float]]:
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    new_data = []
    for row in data:
        row_data, utility = row
        board, fill_num, prev_local_action = row_data
        state = State(board=board, fill_num=fill_num, prev_local_action=prev_local_action)
        new_data.append((state, utility))
    return new_data
