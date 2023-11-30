import copy


class Player:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class State:
    def __init__(self, board: list) -> None:
        self.board = board
        self.lines = self.get_lines()

    def get_lines(self) -> list:
        return [
            [self.board[0][0], self.board[0][1], self.board[0][2]],
            [self.board[1][0], self.board[1][1], self.board[1][2]],
            [self.board[2][0], self.board[2][1], self.board[2][2]],
            [self.board[0][0], self.board[1][0], self.board[2][0]],
            [self.board[0][1], self.board[1][1], self.board[2][1]],
            [self.board[0][2], self.board[1][2], self.board[2][2]],
            [self.board[0][0], self.board[1][1], self.board[2][2]],
            [self.board[0][2], self.board[1][1], self.board[2][0]],
        ]

    def evaluate_line(self, line: list) -> int:
        max_count = line.count("X")
        min_count = line.count("O")
        empty_count = line.count(" ")

        if max_count == 3:
            return 100
        elif min_count == 3:
            return -100

        if max_count == 2 and empty_count == 1:
            return 20
        if min_count == 2 and empty_count == 1:
            return -20

        # if (max_count == 1 and empty_count == 2):
        #     return 10
        # if (min_count == 1 and empty_count == 2):
        #     return -10

        return 0

    def evaluate_func(self) -> int:
        board_evaluation = 0

        for line in self.lines:
            board_evaluation += self.evaluate_line(line)
        return board_evaluation

    def check_if_win(self) -> str or None:
        for line in self.lines:
            if line.count("X") == 3:
                return "X"
            if line.count("O") == 3:
                return "O"
        return None

    def is_terminal(self) -> list:
        return (self.check_if_win()) or " " not in [
            cell for row in self.board for cell in row
        ]

    def display(self) -> None:
        for row in self.board:
            print("|".join(row))


class Game:
    def __init__(
        self,
        player_max: Player,
        player_min: Player,
        x_depth: int,
        o_depth: int,
        is_user_player: bool = False,
    ) -> None:
        self.state = State([[" " for _ in range(3)] for _ in range(3)])
        self.players = {"X": player_max, "O": player_min}
        self.depth_X = x_depth
        self.depth_O = o_depth
        self.is_user = is_user_player

    def is_user_player(self) -> bool:
        return self.is_user

    def play(self, iters: int = 1) -> None:
        current_player = self.players["X"]
        # current_player = self.players["O"]

        print(f"depth_X = {self.depth_X}")
        print(f"depth_O = {self.depth_O}")
        print(f"###_ {iters} _###")
        while not self.state.is_terminal():
            iters += 1
            self.state.display()
            print(f"###_ {iters} _###")

            if current_player == self.players["O"]:
                if self.is_user_player():
                    row = int(input("input row\n"))
                    col = int(input("input col\n"))
                    next_move = row, col
                else:
                    next_move = get_move(
                        current_player.symbol, self.state, self.depth_O
                    )

                self.state = new_board_state(
                    self.state, next_move, current_player.symbol
                )
                current_player = (
                    self.players["O"]
                    if current_player == self.players["X"]
                    else self.players["X"]
                )
                continue

            next_move = get_move(current_player.symbol, self.state, self.depth_X)
            self.state = new_board_state(self.state, next_move, current_player.symbol)
            current_player = self.players[switch_player(current_player.symbol)]

        self.state.display()
        print(iters)
        winner = self.state.check_if_win()
        if winner == "X":
            print(">MAX 'X' wins!")
        elif winner == "O":
            print(">MIN 'O' wins!")
        else:
            print(">Draw!")


def possible_moves(state: State) -> list:
    return [(i, j) for i in range(3) for j in range(3) if state.board[i][j] == " "]


def switch_player(player_symbol: str) -> str:
    if player_symbol == "X":
        return "O"
    else:
        return "X"


def new_board_state(state, next_move: tuple, player_symbol: str) -> State:
    copy_board = copy.deepcopy(state.board)
    new_board = [row for row in copy_board]
    new_board[next_move[0]][next_move[1]] = player_symbol
    return State(new_board)


def get_move(player_symbol: str, state: State, my_depth: int) -> tuple:
    best_utility = minimax(state, player_symbol, my_depth)
    best_move = None
    for next_move in possible_moves(state):
        result_state = new_board_state(state, next_move, player_symbol)
        new_utility = minimax(result_state, switch_player(player_symbol), my_depth - 1)
        if new_utility == best_utility:
            best_move = next_move
            break
    return best_move


def minimax(
    state: State, player_symbol: str, depth, alpha=float("-inf"), beta=float("inf")
) -> int:
    if state.is_terminal() or depth == 0:
        return state.evaluate_func()

    if player_symbol == "X":
        for next_move in possible_moves(state):
            result_state = new_board_state(state, next_move, player_symbol)
            new_utility = minimax(result_state, "O", depth - 1, alpha, beta)
            alpha = max(alpha, new_utility)
            if alpha >= beta:
                return alpha
        return alpha
    else:
        for next_move in possible_moves(state):
            result_state = new_board_state(state, next_move, player_symbol)
            new_utility = minimax(result_state, "X", depth - 1, alpha, beta)
            beta = min(beta, new_utility)
            if alpha >= beta:
                return beta
        return beta


if __name__ == "__main__":
    player_maximizing = Player("X")
    player_minimizing = Player("O")

    game = Game(player_maximizing, player_minimizing, 5, 2)
    game.play()
