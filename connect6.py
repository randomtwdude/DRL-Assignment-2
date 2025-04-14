import sys
import numpy as np
import random
import copy
import time

# UCT Node for MCTS
class UCTNode:
    def __init__(self, candidates, turn, parent=None):
        self.available_moves = set(candidates) # copy
        self.tomove = turn
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.avg = 0

    def uct_rating(self, c):
        return self.avg + c * np.sqrt(np.log(1 + self.parent.visits) / (self.visits + 1e-8))


class UCTMCTS:
    def __init__(self, game,exploration_constant=72.7, rollout_depth=20):
        self.game = game
        self.side = game.turn
        self.c = exploration_constant
        self.rollout_depth = rollout_depth

    def select_child(self, node):
        return max(node.children, key = lambda k: node.children[k].uct_rating(self.c))

    def rollout(self, sim_game, depth):
        current_side = sim_game.turn

        for i in range(depth):
            result = sim_game.check_win()
            if result > 0:
                break

            selected = random.choice(list(sim_game.candidates))
            move_str = f"{sim_game.index_to_label(selected[1])}{selected[0] + 1}"
            sim_game.play_move(current_side, move_str, do_print=False)

            if i % 2:
                current_side = 3 - current_side # switch side

        if result == 0:
            return sim_game.heuristic(self.side)
        elif result == self.side:
            return 100000
        else:
            return -100000

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.avg += (result - node.avg) / node.visits
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_game = copy.deepcopy(self.game)

        # select
        while not node.available_moves:
            move = self.select_child(node)
            move_str = f"{sim_game.index_to_label(move[1])}{move[0] + 1}"
            sim_game.play_move(node.tomove, move_str, do_print=False)
            node = node.children[move]

        # expand
        move = node.available_moves.pop()
        move_str = f"{sim_game.index_to_label(move[1])}{move[0] + 1}"
        sim_game.play_move(node.tomove, move_str, do_print=False)

        new_node = UCTNode(sim_game.candidates, sim_game.turn, node)
        node.children[move] = new_node

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_game, self.rollout_depth)
        # Backpropagation: Update the tree with the rollout reward.
        self.backpropagate(node.children[move], rollout_reward)

    def decide(self, root):
        if len(root.available_moves) == 1:
            return root.available_moves.pop()

        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.candidates = {(15, 15)} # all empty squares around where are stones
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def update_candidates(self, move, radius=2):
        """Call after playing"""
        try:
            self.candidates.remove(move)
        except KeyError:
            pass # it's probably fine
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            for i in range(-radius, radius + 1):
                pos_x = move[1] + i * dx
                pos_y = move[0] + i * dy
                if pos_x < 0 or pos_x >= self.size or pos_y < 0 or pos_y >= self.size:
                    continue
                if self.board[pos_x][pos_y] == 0:
                    self.candidates.add((pos_x, pos_y))

    def heuristic(self, side):
        black, white = self.evaluate_board()
        return (black - white) if side == 1 else (white - black)

    def evaluate_board(self):
        """Use heuristic()!"""
        score_b = score_w = 0
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for x in range(self.size):
            for y in range(self.size):
                for dx, dy in directions:
                    count_b = count_w = 0
                    for i in range(6):
                        nx, ny = x + i * dx, y + i * dy
                        if 0 <= nx < 19 and 0 <= ny < 19:
                            if self.board[nx][ny] == 1: # black
                                count_b += 1
                                count_w = 0
                            elif self.board[nx][ny] == 2: # white
                                count_w += 1
                                count_b = 0
                        else:
                            count_b = count_w = 0
                            break

                    if count_b == 6:
                        score_b += 50000
                    elif count_b == 5:
                        score_b += 5000
                    elif count_b == 4:
                        score_b += 500
                    elif count_b == 3:
                        score_b += 50

                    if count_w == 6:
                        score_w += 50000
                    elif count_w == 5:
                        score_w += 5000
                    elif count_w == 4:
                        score_w += 500
                    elif count_w == 3:
                        score_w += 50

        return score_b, score_w

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move, do_print=True):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        if color == 1:
            color = 'b'
        elif color == 2:
            color = 'w'

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        if do_print:
            print('= ', end='', flush=True)

        # transform "Q16" (label(move[1])move[0]+1) back to (15, 15)
        move = (int(move[1:]) - 1, self.label_to_index(move[0]))
        self.update_candidates(move)

    def generate_move(self, color):
        """MCTS"""
        if self.game_over:
            print("? Game over")
            return

        mcts = UCTMCTS(self)
        root = UCTNode(self.candidates, self.turn)
        if len(root.available_moves) > 1:
            start_time = time.time()
            while time.time() - start_time < 7: # just run for 7 seconds
                mcts.run_simulation(root)
        selected = mcts.decide(root)

        move_str = f"{self.index_to_label(selected[1])}{selected[0] + 1}"
        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)
        return

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return

        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
