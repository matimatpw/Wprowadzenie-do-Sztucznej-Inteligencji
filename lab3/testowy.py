from enum import Enum

class Reward(Enum):
    X_WIN = 1
    Y_WIN = -1
    DRAW  = 0

class Player:
    def __init__(self, symbol):
        self.symbol = symbol

class State:
    def __init__(self, board):
        self.board = board
        self.lines = self.get_lines()

    def get_lines(self):
        return [
        [self.board[0][0], self.board[0][1], self.board[0][2]],
        [self.board[1][0], self.board[1][1], self.board[1][2]],
        [self.board[2][0], self.board[2][1], self.board[2][2]],
        [self.board[0][0], self.board[1][0], self.board[2][0]],
        [self.board[0][1], self.board[1][1], self.board[2][1]],
        [self.board[0][2], self.board[1][2], self.board[2][2]],
        [self.board[0][0], self.board[1][1], self.board[2][2]],
        [self.board[0][2], self.board[1][1], self.board[2][0]] 
        ]

    def evaluate_line(self, line:list):
        max_count = line.count('X')
        min_count = line.count('O')
        empty_count = line.count(' ')

        if(max_count == 3):
            return 100
        elif(min_count == 3):
            return -100
        
        if (max_count == 2 and empty_count == 1):
            return 10
        if (min_count == 2 and empty_count == 1):
            return -10
        
        if (max_count == 1 and empty_count == 2):
            return 1
        if (min_count == 1 and empty_count == 2):
            return -1

        return 0
        


    def evaluate_func(self):
        board_evaluation = 0

        for line in self.lines:
            board_evaluation += self.evaluate_line(line)
        
        return board_evaluation

    def check_if_win(self):
        for line in self.lines:
            if(line.count('X') == 3):
                return 'X'
            if(line.count('O') == 3):
                return 'O'
        return None

    def is_terminal(self):
        # Check if the game is over (either someone wins or it's a draw)
        return (self.check_if_win()) or ' ' not in [cell for row in self.board for cell in row]


    def utility(self):
        pass

    def display(self):
        # Display the current state of the game
        for row in self.board:
            print("|".join(row))
            print("-----")

class Game:
    def __init__(self, player_X, player_O, my_depth=10):
        self.state = State([[' ' for _ in range(3)] for _ in range(3)])  # Initial game state
        self.players = {'X': player_X, 'O': player_O}
        self.depth = my_depth

    def play(self):
        # current_player = self.players["X"]
        current_player = self.players["O"]
        iters = 1

        print(f"depth= {self.depth}")
        print(f"###_ {iters} _###")
        while not self.state.is_terminal():
            iters += 1
            self.state.display()
            print(f"###_ {iters} _###")
            action = get_action(current_player.symbol, self.state, self.depth)
            self.state = result(self.state, action, current_player.symbol)
            current_player = self.players['O'] if current_player == self.players['X'] else self.players['X']

        self.state.display()
        print(iters)
        winner = self.state.check_if_win()
        if winner == 'X':
            print("Player MAX 'X' wins!")
        elif winner == 'O':
            print("Player MIN 'O' wins!")
        else:
            print("It's a draw!")

def actions(state):
    # Return a list of possible moves
    return [(i, j) for i in range(3) for j in range(3) if state.board[i][j] == ' ']

def result(state, action, player_symbol):
    # Return a new state after a player makes a move
    new_board = [row.copy() for row in state.board]
    new_board[action[0]][action[1]] = player_symbol
    return State(new_board)


def get_action(player_symbol, state, my_depth):
    _, action = minimax(state, player_symbol, my_depth)
    return action

#TODO alpfa beta pruning | evaluate function | enumerate zastosowac

def minimax(state:State, player, depth):
    if state.is_terminal() or depth == 0:
        return state.evaluate_func(), None

    if player == 'X':
        max_utility = float('-inf')
        best_action = None
        for action in actions(state):
            result_state = result(state, action, player)
            utility, _ = minimax(result_state, 'O', depth - 1)
            if utility > max_utility:
                max_utility = utility
                best_action = action
        return max_utility, best_action
    else:
        min_utility = float('inf')
        best_action = None
        for action in actions(state):
            result_state = result(state, action, player)
            utility, _ = minimax(result_state, 'X', depth - 1)
            if utility < min_utility:
                min_utility = utility
                best_action = action
        return min_utility, best_action

if __name__ == "__main__":
    player_X = Player('X')
    player_O = Player('O')

    game = Game(player_X, player_O,10)

    # Start move center #
    game.state.board[1][1] = "X"
    game.state.board[0][0] = "X"
    game.state.board[0][1] = "O"

    game.play()
