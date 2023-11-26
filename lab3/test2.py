class Player:
    def __init__(self, symbol):
        self.symbol = symbol

class State:
    def __init__(self, board):
        self.board = board

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

    def display(self):
        # Wizualizujemy stan gry
        for row in self.board:
            print("|".join(row))
            print("-----")

class Game:
    def __init__(self, player_X, player_O):
        self.state = State([[' ']*3 for _ in range(3)])  # Początkowy stan gry
        self.players = {'X': player_X, 'O': player_O}

    def play(self):
        current_player = 'X'
        while not self.state.is_terminal():
            self.state.display()
            print(f"Gracz {current_player}, twój ruch:")
            action = self.players[current_player].get_action(self.state)
            self.state = result(self.state, action, current_player)
            current_player = 'O' if current_player == 'X' else 'X'

        self.state.display()
        winner = self.state.utility()
        if winner == 1:
            print("Gracz X wygrywa!")
        elif winner == -1:
            print("Gracz O wygrywa!")
        else:
            print("Remis!")

def actions(state):
    # Zwraca listę możliwych ruchów
    return [(i, j) for i in range(3) for j in range(3) if state.board[i][j] == ' ']

def result(state, action, player):
    # Zwraca nowy stan po wykonaniu ruchu przez gracza
    new_board = [row.copy() for row in state.board]
    new_board[action[0]][action[1]] = player
    return State(new_board)

class HumanPlayer:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_action(self, state):
        while True:
            try:
                row = int(input("Podaj numer wiersza (0-2): "))
                col = int(input("Podaj numer kolumny (0-2): "))
                if state.board[row][col] == ' ':
                    return row, col
                else:
                    print("To pole jest już zajęte. Wybierz inne.")
            except ValueError:
                print("Nieprawidłowe dane. Podaj liczby.")

if __name__ == "__main__":
    player_X = HumanPlayer('X')
    player_O = HumanPlayer('O')
    
    game = Game(player_X, player_O)
    game.play()
