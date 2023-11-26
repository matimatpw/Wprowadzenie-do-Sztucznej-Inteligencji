
board = [[' ' for _ in range(3)] for _ in range(3)]
def get_lines():
        lines = []
        row_lines = []
        col_lines = []
        for row in range(3):
            for col in range(3):
                row_lines.append([board[row][col]])
                col_lines.append(board[col][row])
            lines.append(row_lines)
            lines.append(col_lines)
        print(lines)
get_lines()