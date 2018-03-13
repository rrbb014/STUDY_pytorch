# Tic-Tac-Toe Environment
# author: rrbb014
# date: 2018.03.02
# reference: Reinforcement Learning: An introduction, 2nd edition
#            https://github.com/ShangtongZhang/reinforcement-learning-an-introduction

import os
import numpy as np
from dotenv import load_dotenv
from itertools import product
from collections import defaultdict, namedtuple

load_dotenv('./.env')

state_tuple = namedtuple('state_tuple', 'state, end, value, reward')

class State:

    def __init__(self):
        self.width = int(os.environ.get('ENV_WIDTH'))
        self.height = int(os.environ.get('ENV_HEIGHT'))
        self.data = np.zeros([self.width, self.height])
        self.winner = None
        self.hash_value = None
        self.end = None

    def get_hash_value(self):
        if self.hash_value == None:
            self.hash_value = 0
            for i in self.data.reshape(self.width * self.height):
                if i == -1:
                    i = 2
                self.hash_value = self.hash_value * 3 + i
        self.hash_value = int(self.hash_value)
        return int(self.hash_value)
        
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    def check_game(self):
        if self.end is not None:
            return self.end
        results = []
        # Check Rows
        for r in self.data:
            results.append(r.sum())
        # Check Columns
        for c in self.data.T:
            results.append(c.sum())
        # Check diagonal
        diagonal = self.data.diagonal().sum()
        # Check reversed diagonal
        reverse_diagonal = 0
        for r in range(self.width):
            reverse_diagonal += self.data[r][self.width-r-1]

        results.append(diagonal)
        results.append(reverse_diagonal)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end

            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # 비길때
        if np.sum(np.abs(self.data)) == self.width * self.height:
            self.winner = 0
            self.end = True
            return self.end

        self.end = False
        return self.end
        
        
class TictactoeEnv:

    def __init__(self):
        self.width = int(os.environ.get('ENV_WIDTH'))
        self.height = int(os.environ.get('ENV_HEIGHT'))
        self.gameboard = np.zeros([self.width, self.height])
        self.state_transition_probability = os.environ.get('STATE_TRANSITION_PROBABILITY')
        self.possible_actions = range(9)
        #self.possible_actions = list(product(range(self.width), range(self.height)))
        self.all_states = defaultdict(lambda : 0)
        self._get_all_states()

    def __str__(self):
        change_symbol_dict = {0: '0', 1: '*', -1: 'X'}
        return '''
            |  {0}  |  {1}  |  {2}  |
            -------------------
            |  {3}  |  {4}  |  {5}  |
            -------------------
            |  {6}  |  {7}  |  {8}  |
            '''.format(change_symbol_dict[self.gameboard[0][0]], change_symbol_dict[self.gameboard[0][1]], change_symbol_dict[self.gameboard[0][2]],
                                                change_symbol_dict[self.gameboard[1][0]], change_symbol_dict[self.gameboard[1][1]], change_symbol_dict[self.gameboard[1][2]],
                                                change_symbol_dict[self.gameboard[2][0]], change_symbol_dict[self.gameboard[2][1]], change_symbol_dict[self.gameboard[2][2]])

    def _get_all_states_implementation(self, current_state, current_symbol, all_states):
        for i in range(0, self.width):
            for j in range(0, self.height):
                if current_state.data[i][j] == 0:
                    new_state = current_state.next_state(i, j, current_symbol)
                    new_hash = new_state.get_hash_value()
                    if new_hash not in all_states.keys():
                        is_end = new_state.check_game()
                        if is_end:
                            if new_state.winner == 1:
                                reward = (5, 0)
                            elif new_state.winner == -1:
                                reward = (0, 5)
                            else:
                                reward = (1, 1)
                        else:
                            reward = (0, 0)
                        all_states[new_hash] = state_tuple(new_state, is_end, 0, reward)
                        if not is_end:
                            self._get_all_states_implementation(new_state, -current_symbol, all_states)
        return all_states

    def _get_all_states(self):
        current_symbol = 1
        current_state = State()
        all_states = dict()
        all_states[current_state.get_hash_value()] = state_tuple(current_state, current_state.check_game(), 0, (0, 0))
        all_states = self._get_all_states_implementation(current_state, current_symbol, all_states)
        self.all_states = all_states

    def get_all_states(self):
        return self.all_states

    def check_game(self):
        env = self.gameboard
        diagonal = env.diagonal().sum()
        reverse_diagonal = 0
        for r in range(env.width):
            reverse_diagonal += env[r][env.width-r-1]
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

    def state_after_action(self, state, action, symbol):
        state_obj = State()
        current_state = state.copy()
        current_action = action
        current_action -= 1
        i = current_action // self.height
        j = current_action % self.height
        if current_state[i][j] == 0:
            current_state[i][j] = symbol
            state_obj.data = current_state
            state_obj.get_hash_value()
            return state_obj
        else:
            return False
    
    def get_reward(self, state_obj, symbol):
        position = 0 if symbol == 1 else 1
        if state_obj.hash_value is None:
            state_obj.get_hash_value()
        return self.all_states[state_obj.hash_value].reward[position]