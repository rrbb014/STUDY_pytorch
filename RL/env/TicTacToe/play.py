# Tic-Tac-Toe playing source code
# author: rrbb014
# date: 2018.03.02

from environment import TictactoeEnv
from agents import HumanPlayer

if __name__ == '__main__':
	env = TictactoeEnv()
	player1 = HumanPlayer('player1', 1, env)
	player2 = HumanPlayer('player2', -1, env)
	
	while True:
		print(env)
		# Player1 turn
		player1.action()
		if env.check_game():
			break
		print(env)
		player2.action()
		if env.check_game():
			break 