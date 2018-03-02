# Tic-Tac-Toe agents Implementation
# Author: rrbb014
# date: 2018.02.23
# reference: Reinforcement Learning: An introduction, 2nd edition
#            https://github.com/ShangtongZhang/reinforcement-learning-an-introduction


class HumanPlayer:

	def __init__(self, name, symbol, env):
		self.name = name
		self.symbol = symbol
		self.current_state = None
		self.current_action = None
		self.env = env

	def action(self):
		while True:
			self.current_action = int(input("%s 's turn...Please type your position >> " % self.name))
			if self.env.check_input(self):
				break
			else:
				print('Wrong choice! Try again..')