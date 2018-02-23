# Tic-Tac-Toe Implementation
# Author: rrbb014
# date: 2018.02.23
# reference: Reinforcement Learning: An introduction, 2nd edition
#            https://github.com/ShangtongZhang/reinforcement-learning-an-introduction

import os
import numpy as np
from dotenv import load_dotenv

load_dotenv('./.env')

class TictactoeEnvironment:
	
	def __init__(self):
		self.width = int(os.environ.get('ENV_WIDTH'))
		self.height = int(os.environ.get('ENV_HEIGHT'))
		self.gameboard = np.zeros([self.width, self.height])
		
	
	def __str__(self):
		change_symbol_dict = {0: '0', 1: '*', -1: 'X'}
		return '''
			|  {0}  |  {1}  |  {2}  |
			-------------------
			|  {3}  |  {4}  |  {5}  |
			-------------------
			|  {6}  |  {7}  |  {8}  |'''.format(change_symbol_dict[self.gameboard[0][0]], change_symbol_dict[self.gameboard[0][1]], change_symbol_dict[self.gameboard[0][2]],
												change_symbol_dict[self.gameboard[1][0]], change_symbol_dict[self.gameboard[1][1]], change_symbol_dict[self.gameboard[1][2]],
												change_symbol_dict[self.gameboard[2][0]], change_symbol_dict[self.gameboard[2][1]], change_symbol_dict[self.gameboard[2][2]])

	def check_game(self):
		env = self.gameboard
		diagonal = env.diagonal().sum() 
		reverse_diagonal = env[0][2] + env[1][1] + env[2][0]
		results = []
		
		for r in env:
			results.append(r.sum())
		
		for c in env.T:
			results.append(c.sum())
			
		results.append(diagonal)
		results.append(reverse_diagonal)

		for result in results:
			if result == 3:
				print('Win Player1')
				return True
			if result == -3:
				print('Win Player2')
				return True
				
		# 비길때
		if np.sum(np.abs(env)) == 9:
			print('Tied!')
			return True
		
		return False
		
	def check_input(self, player):
		env = self.gameboard
		player.current_action -= 1
		i = player.current_action // self.height
		j = player.current_action % self.height
		if env[i][j] == 0:
			self.gameboard[i][j] = player.symbol
			return True
		else:
			return False
	
		
class HumanPlayer:
	
	def __init__(self, name, symbol):
		self.name = name
		self.symbol = symbol
		self.current_state = None
		self.current_action = None
	
	def action(self):
		while True:
			self.current_action = int(input("Please type your position >> "))
			if env.check_input(self):
				break
			else:
				print('Wrong choice! Try again..')
	
		
if __name__ == '__main__':
	env = TictactoeEnvironment()
	player1 = HumanPlayer('player1', 1)
	player2 = HumanPlayer('player2', -1)
	
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
		