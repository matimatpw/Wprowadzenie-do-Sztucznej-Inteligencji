class Player:
    def __init__(self, symbol):
        self.symbol = symbol

class State:
    def __init__(self, board):
        self.board = board

    def is_terminal(self):
        # Check if the game is over (either someone wins or it's a draw)
        return (self.utility() != 0) or ' ' not in [cell for row in self.board for cell in row]

    def utility(self):
        # Calculate the utility of the state (1 - win, 0 - draw, -1 - loss)
        for row in self.board:
            if row[0] == row[1] == row[2] and row[0] != ' ':
                return 1 if row[0] == 'X' else -1

        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] != ' ':
                return 1 if self.board[0][col] == 'X' else -1
        #diagonally
        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != ' ':
            return 1 if self.board[0][0] == 'X' else -1

        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != ' ':
            return 1 if self.board[0][2] == 'X' else -1

        #full board -> draw
        if(' ' not in [cell for row in self.board for cell in row]):
            return 0

        #heurystyka !!!
        return 0

    def display(self):
        # Display the current state of the game
        for row in self.board:
            print("|".join(row))
            print("-----")

class Game:
    def __init__(self, player_X, player_O):
        self.state = State([[' ' for _ in range(3)] for _ in range(3)])  # Initial game state
        self.players = {'X': player_X, 'O': player_O}

    def play(self):
        current_player = 'X'
        iters = 0
        while not self.state.is_terminal():
            iters += 1
            self.state.display()
            print()
            action = self.players[current_player].get_action(self.state)
            self.state = result(self.state, action, current_player)
            current_player = 'O' if current_player == 'X' else 'X'

        self.state.display()
        print(iters)
        winner = self.state.utility()
        if winner == 1:
            print("Player X wins!")
        elif winner == -1:
            print("Player O wins!")
        else:
            print("It's a draw!")

def actions(state):
    # Return a list of possible moves
    return [(i, j) for i in range(3) for j in range(3) if state.board[i][j] == ' ']

def result(state, action, player):
    # Return a new state after a player makes a move
    new_board = [row.copy() for row in state.board]
    new_board[action[0]][action[1]] = player
    return State(new_board)

class MinMaxPlayer(Player):
    def get_action(self, state):
        _, action = self.minimax(state, self.symbol)
        return action

    def minimax(self, state, player):
        if state.is_terminal():
            return state.utility(), None

        if player == 'X':
            max_utility = float('-inf')
            best_action = None
            how_fast = 0
            for action in actions(state):
                how_fast += 1
                result_state = result(state, action, player)
                utility, _ = self.minimax(result_state, 'O')
                if utility >= max_utility:
                    max_utility = utility
                    best_action = action
            return max_utility, best_action
        else:
            min_utility = float('inf')
            best_action = None
            how_fast = 0
            for action in actions(state):
                how_fast +=1
                result_state = result(state, action, player)
                utility, _ = self.minimax(result_state, 'X')
                if utility <= min_utility:
                    min_utility = utility
                    best_action = action
            return min_utility, best_action

def evaluate(board):
    # Define winning combinations (8 lines: 3 rows, 3 columns, 2 diagonals)
    lines = [
        [board[0][0], board[0][1], board[0][2]],  # Row 1
        [board[1][0], board[1][1], board[1][2]],  # Row 2
        [board[2][0], board[2][1], board[2][2]],  # Row 3
        [board[0][0], board[1][0], board[2][0]],  # Column 1
        [board[0][1], board[1][1], board[2][1]],  # Column 2
        [board[0][2], board[1][2], board[2][2]],  # Column 3
        [board[0][0], board[1][1], board[2][2]],  # Diagonal 1
        [board[0][2], board[1][1], board[2][0]]   # Diagonal 2
    ]

    score = 0

    # Check each line for scores
    for line in lines:
        score += evaluate_line(line)

    return score

def evaluate_line(line):
    computer_count = line.count('X')
    opponent_count = line.count('O')
    empty_count = line.count(' ')

    # Check for 3-in-a-line
    if computer_count == 3:
        return 100
    elif opponent_count == 3:
        return -100

    # Check for 2-in-a-line
    if computer_count == 2 and empty_count == 1:
        return 10
    elif opponent_count == 2 and empty_count == 1:
        return -10

    # Check for 1-in-a-line
    if computer_count == 1 and empty_count == 2:
        return 1
    elif opponent_count == 1 and empty_count == 2:
        return -1

    # No score for this line
    return 0

# Example usage:
board_example = [
    ['X', 'O', 'X'],
    [' ', 'X', 'O'],
    ['O', ' ', 'X']
]

result = evaluate(board_example)
print(f"Evaluation Score: {result}")