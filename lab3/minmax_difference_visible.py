
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
            return 20
        if (min_count == 2 and empty_count == 1):
            return -20

        # if (max_count == 1 and empty_count == 2):
        #     return 20
        # if (min_count == 1 and empty_count == 2):
        #     return -20

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
        return (self.check_if_win()) or ' ' not in [cell for row in self.board for cell in row]

    def display(self):
        for row in self.board:
            print("|".join(row))

class Game:
    def __init__(self, player_max, player_min, my_depth=10):
        self.state = State([[' ' for _ in range(3)] for _ in range(3)])
        self.players = {'X': player_max, 'O': player_min}
        self.depth = my_depth

    def play(self):
        # current_player = self.players["X"]
        current_player = self.players["O"]
        iters = 1

        print(f"depth= {self.depth}")
        print(f"###_ {iters} _###")
        while not self.state.is_terminal():
            if(current_player == self.players['O']):
                iters += 1
                self.state.display()
                print(f"###_ {iters} _###")
                next_move = get_action(current_player.symbol, self.state, 10)
                self.state = new_board_state(self.state, next_move, current_player.symbol)
                current_player = self.players['O'] if current_player == self.players['X'] else self.players['X']
                # self.state.display()
                # row =int(input("input row\n"))
                # col =int(input("input col\n"))
                # action = row,col
                # self.state = new_board_state(self.state, action, current_player.symbol)
                # current_player = self.players['O'] if current_player == self.players['X'] else self.players['X']
                # iters +=1
                # print(f"###_ {iters} _###")
                continue

            iters += 1
            self.state.display()
            print(f"###_ {iters} _###")
            next_move = get_action(current_player.symbol, self.state, 2)
            self.state = new_board_state(self.state, next_move, current_player.symbol)
            current_player = self.players['O'] if current_player == self.players['X'] else self.players['X']

        self.state.display()
        print(iters)
        winner = self.state.check_if_win()
        if winner == 'X':
            print(">MAX 'X' wins!")
        elif winner == 'O':
            print(">MIN 'O' wins!")
        else:
            print(">Draw!")

def possible_moves(state):
    return [(i, j) for i in range(3) for j in range(3) if state.board[i][j] == ' ']

def new_board_state(state, next_move, player_symbol):
    # Return a new state after a player makes a move
    new_board = [row.copy() for row in state.board]
    new_board[next_move[0]][next_move[1]] = player_symbol
    return State(new_board)


def get_action(player_symbol, state, my_depth):
    _, best_move = minimax(state, player_symbol, my_depth)
    return best_move

#TODO alpfa beta pruning

def minimax(state:State, player_symbol, depth):
    if state.is_terminal() or depth == 0:
        if(possible_moves(state)):
            move = possible_moves(state)[0]
        else:
            move = None
        return state.evaluate_func(), None


    if player_symbol == 'X':
        max_utility = float('-inf')
        best_move = None
        for next_move in possible_moves(state):
            result_state = new_board_state(state, next_move, player_symbol)
            utility, _ = minimax(result_state, 'O', depth -1)
            if utility > max_utility:
                max_utility = utility
                best_move = next_move
        return max_utility, best_move
    else:
        min_utility = float('inf')
        best_move = None
        for next_move in possible_moves(state):
            result_state = new_board_state(state, next_move, player_symbol)
            utility, _ = minimax(result_state, 'X', depth -1)
            if utility < min_utility:
                min_utility = utility
                best_move = next_move
        return min_utility, best_move

if __name__ == "__main__":
    player_maximizing = Player('X')
    player_minimizing = Player('O')

    game = Game(player_maximizing, player_minimizing,30)

    #TODO Start move center # PLANSZA POCZATKOWA DO SPRAWKA bo wtedy wieksza glebokosc sprawia ze dluzej gra trwa
    # game.state.board[0][0] = "X"
    # game.state.board[1][1] = "O"
    # game.state.board[0][2] = "X"
    # game.state.board[1][1] = "O"
    # game.state.board[2][2] = "X"

    game.play()
