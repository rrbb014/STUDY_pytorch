# Tic-Tac-Toe agents Implementation
# Author: rrbb014
# date: 2018.02.23
# reference: Reinforcement Learning: An introduction, 2nd edition
#            https://github.com/ShangtongZhang/reinforcement-learning-an-introduction
#            파이썬과 케라스로 만나는 강화학습 
import os
import dill
import random
from collections import defaultdict, namedtuple
from dotenv import load_dotenv
from environment import TictactoeEnv


load_dotenv('./.env')


class PolicyIterationAgent():
    
    def __init__(self, symbol, env):
        self.symbol = symbol
        self.first_play = True if self.symbol == 1 else False
        self.env = env
        self.discount_factor = float(os.environ.get('DISCOUNT_FACTOR'))
        self.policy_table = defaultdict(lambda : [1 / len(env.possible_actions)] * len(env.possible_actions))
        self.value_table = defaultdict(lambda : 0)
        
    def policy_evaluation(self):
        
        all_state_dict = self.env.all_states.copy()
        next_value_table = defaultdict(lambda : 0)
        
        for state_key in all_state_dict:
            value = 0.
            # End 상태의 가치 함수 = 0
            if all_state_dict[state_key].end:
                next_value_table[state_key] = 0
                continue
            if self.first_play:
                if sum(all_state_dict[state_key].state.data.reshape(self.env.width * self.env.height)) == 0:
                    for action in self.env.possible_actions:
                        next_state = self.env.state_after_action(all_state_dict[state_key].state.data, action + 1, self.symbol)
                        if next_state:
                            reward = self.env.get_reward(next_state, self.symbol)
                            next_value = self.get_value(next_state)
                            value += self.get_policy(all_state_dict[state_key].state)[action] * (reward + (self.discount_factor * next_value))
                    next_value_table[state_key] = round(value, 2)

                else:
                    continue
            else:
                if sum(all_state_dict[state_key].state.data.reshape(self.env.width * self.env.height)) == 1:
                    for action in self.env.possible_actions:
                        next_state = self.env.state_after_action(all_state_dict[state_key].state.data, action + 1, self.symbol)
                        if next_state:
                            reward = self.env.get_reward(next_state, self.symbol)
                            next_value = self.get_value(next_state)
                            value += self.get_policy(all_state_dict[state_key].state)[action] * (reward + (self.discount_factor * next_value))
                    next_value_table[state_key] = round(value, 2)

                else:
                    continue
        # update value function
        self.value_table = next_value_table
    
    def policy_improvement(self):

        all_state_dict = self.env.all_states.copy()
        next_policy_table = self.policy_table.copy()
        
        for state_key in all_state_dict:
            if all_state_dict[state_key].end:
                continue
            value = -99999
            max_index = []
            # 반환할 정책 초기화
            result = [0.] * len(self.env.possible_actions)
            
            # 모든 행동에 대해 reward + (discount_factor * next_value) 계산
            # 가치가 max인 action을 찾아라! (복수면 모두)
            if self.first_play:
                if sum(all_state_dict[state_key].state.data.reshape(self.env.width * self.env.height)) == 0:
                    for index, action in enumerate(self.env.possible_actions):
                        next_state = self.env.state_after_action(all_state_dict[state_key].state.data, action+1, self.symbol)
                        if next_state:
                            reward = self.env.get_reward(next_state, self.symbol)
                            next_value = self.get_value(next_state)
                            temp = reward + self.discount_factor * next_value

                            # 보상이 최대인 행동 index를 모두 추출
                            if temp == value:
                                max_index.append(index)
                            elif temp > value:
                                value = temp
                                max_index.clear()
                                max_index.append(index)
                else:
                    continue
            else:
                if sum(all_state_dict[state_key].state.data.reshape(self.env.width * self.env.height)) == 1:
                    for index, action in enumerate(self.env.possible_actions):
                        next_state = self.env.state_after_action(all_state_dict[state_key].state.data, action+1, self.symbol)
                        if next_state:
                            reward = self.env.get_reward(next_state, self.symbol)
                            next_value = self.get_value(next_state)
                            temp = reward + self.discount_factor * next_value

                            # 보상이 최대인 행동 index를 모두 추출
                            if temp == value:
                                max_index.append(index)
                            elif temp > value:
                                value = temp
                                max_index.clear()
                                max_index.append(index)
                else:
                    continue
            prob = 1 / len(max_index)
        
            for index in max_index:
                result[index] = prob

            next_policy_table[state_key] = result
        
        # Update lastly
        self.policy_table = next_policy_table
        
    def get_policy(self, state):
        if state.end:
            return 0.0
        return self.policy_table[state.hash_value]
    
    def get_value(self, state):
        return self.value_table[state.hash_value]
    
    def action(self):
	
    def get_action(self, state):
        # policy에 들어있는 행동 중 무작위로 하나의 행동을 추출
        # k - parameter를 수정하면 복수개의 행동 추출 가능
        # weights 파라미터가 정책분포 - policy variable
        policy = self.get_policy(state)
        action_index = random.choices(env.possible_actions, weights=policy, k=1)[0]
        return action_index
    
    def save_model(self):
        if len(self.policy_table) != 0 and len(self.value_table) != 0:
            with open('./agents_model/policy_iteration_policy_' + str(self.symbol), 'wb') as f:
                dill.dump(self.policy_table, f)
            with open('./agents_model/policy_iteration_value_' + str(self.symbol), 'wb') as f:
                dill.dump(self.value_table, f)
        else:
            print('No data. please explore.')
    
    def load_model(self):
        try:
            with open('./agents_model/policy_iteration_policy_' + str(self.symbol), 'rb') as f:
                self.policy_table = dill.load(f)
            with open('./agents_model/policy_iteration_value_' + str(self.symbol), 'rb') as f:
                self.value_table = dill.load(f)
        except(FileNotFoundError):
            print('No model files. Cannot load model')


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
