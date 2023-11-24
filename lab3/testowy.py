

class Player:
    def __init__(self, symbol:str) -> None:
        self.symbol = symbol.upper()

    def make_move(self):
        """
        >>> bool(" ")
        True
        """
        pass

class Game:
    def __init__(self) -> None:
        self.board = [str(_) for _ in range(9)]
        pass

    def display(self):
        row = ""
        for idx, place in enumerate(self.board):
            end_line = " | "
            if((idx % 3)):
                if(idx % 3 == 2):
                    end_line = ""
                row += (str(self.board[idx]) +end_line)
            else:
                row += ("\n" + str(self.board[idx]) +end_line)
        print(row)

    def place_move(self, move_idx, my_player: Player):
        self.board[move_idx] = my_player.symbol








class State:
    def __init__(self, my_game: Game) -> None:
        self.game = my_game


    @staticmethod
    def get_empty_field(self, my_game:Game):
        empty = []
        for idx, field in enumerate(my_game.board):
            if(field == "X" or field == "O"):
                empty.append(idx)
        return empty

def main():
    gm = Game()
    gm.display()

main()