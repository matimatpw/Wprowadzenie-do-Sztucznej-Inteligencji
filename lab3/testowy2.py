class Player:
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol

class GameState:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = None

    def is_terminal(self):
        # Sprawdzamy, czy gra została zakończona (czy jest zwycięzca lub remis)
        return self.utility() != 0 or ' ' not in self.board

    def utility(self):
        # Obliczamy użyteczność stanu (1 - wygrana, 0 - remis, -1 - przegrana)
        for row in range(3):
            if self.board[row][0] == self.board[row][1] == self.board[row][2] != ' ':
                return 1 if self.board[row][0] == 'X' else -1

        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return 1 if self.board[0][col] == 'X' else -1

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return 1 if self.board[0][0] == 'X' else -1

        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return 1 if self.board[0][2] == 'X' else -1

        return 0


    def get_empty_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']

    def print_board(self):
        for row in self.board:
            print('|'.join(row))
            print('-' * 5)

    def is_board_full(self):
        for row in self.board:
            if ' ' in row:
                return False
        return True

    def check_winner(self):
        # Check rows
        for row in self.board:
            if all(cell == row[0] and cell != ' ' for cell in row):
                return True

        # Check columns
        for col in range(3):
            if all(self.board[row][col] == self.board[0][col] and self.board[row][col] != ' ' for row in range(3)):
                return True

        # Check diagonals
        if all(self.board[i][i] == self.board[0][0] and self.board[i][i] != ' ' for i in range(3)):
            return True
        if all(self.board[i][2 - i] == self.board[0][2] and self.board[i][2 - i] != ' ' for i in range(3)):
            return True

        return False

class Game:
    def __init__(self, maxiizing, minimizing):
        self.players = [Player(maxiizing, 'X'), Player(minimizing, 'O')]
        self.game_state = GameState()
        self.current_turn = 0

    # def switch_player(self):
    #     self.current_turn = (self.current_turn + 1) % 2
    #     self.game_state.current_player = self.players[self.current_turn]

    # def place_move(self, row, col):
    #     self.game_state.board[row][col] = self.game_state.current_player.symbol




    def play(self):
        self.game_state.current_player = self.players[self.current_turn]

        while True:
            self.game_state.print_board()

            row = int(input(f"{self.game_state.current_player.name}, enter row (0, 1, or 2): "))
            col = int(input("Enter column (0, 1, or 2): "))

            if self.game_state.board[row][col] == ' ':
                self.game_state.board[row][col] = self.game_state.current_player.symbol

                if self.game_state.check_winner():
                    self.game_state.print_board()
                    print(f"Congratulations, {self.game_state.current_player.name}! You won!")
                    break
                elif self.game_state.is_board_full():
                    self.game_state.print_board()
                    print("It's a tie!")
                    break
                else:
                    self.switch_player()
            else:
                print("Cell already occupied. Try again.")
    def test(self):
        print(self.game_state.get_empty_moves())

# Przykład użycia
if __name__ == "__main__":
    player1_name = "Max"
    player2_name = "Min"

    game = Game(player1_name, player2_name)
    game.
    game.test()

