import numpy as np
import dqn as D
import tdrl as T
from matplotlib import pyplot as plt
from collections import defaultdict

'''
Explore RL vs Deep RL comparison???

Types of firms
1. Lifetime Nash punishment strategy (closed form). Knows N,ci. assumes symmetric cj's.
2. Bayesian Learning firm. Learns N, Learns cj's. Learns demand transitions. Lifetime value maximization.


Other components of utility function besides profits
1. Entry and Exit costs. Periodic quantity decision made simultaneously.

Observations
1. All firms of type (3) product the pareto optimal outcome.
2. All firms of type (4) should theoritically achive this strategy. But as N increases?
3. How do type (5) firm compare with type (3,4)?
4. What happens with competing deepRL firms having different model complexity?
'''

Q = 8
q_max = 8.0
EPISODES = 1

class action_space():
	
	q_min = 0.0
	q_max = 100.0  # Not a strict bound
	N = None
	
	def __init__(self,N):
		self.N = N
		self.q_max = q_max
	
	def check(self,A):
		if np.sum(A < self.q_min) > 0:
			return False
		return True
	
	def sample(self):
		return np.random.uniform(self.q_min, self.q_max, size = (self.N,))

class market_entry_exit_game():
	'''
	This class models the environment by generating rewards and 
	new state given an action and initial state.
	'''

	demand = None
	S = None
	action_space = None
	N = None
	action_history = None
	T = 1000000
	
	def __init__(self, players):
		self.N = len(players)
		self.demand = q_max
		self.S = np.concatenate((np.zeros(self.N),[self.demand]))
		self.C = map(lambda x: x.c, players)
		for player in players:
			if hasattr(player, 'N'):
				player.N = self.N
			if hasattr(player, 'C_competitors'):
				player.C_competitors = sum(self.C) - player.c
		self.action_space = action_space(self.N)
		self.action_history = np.empty((1,self.N))

	def solve_demand_supply(self):
		'''
		Assumes a linear inverse price-demand curve.
		Solves for any number of agents
		'''
		p = self.S[-1] - np.sum(self.S[:-1])
		profits = map(lambda c,q: (p - c)*q, self.C,self.S[:-1])
		return profits
	
	def step(self,A):
		self.T -= 1
		S_new = np.zeros(shape=self.S.shape)
		S_new[:-1] = self.firm_state_transition(A)
		S_new[-1] = self.demand_transition()
		self.S = S_new
		profits = self.solve_demand_supply()
		self.action_history = np.vstack([self.action_history,A])
		return self.S, profits, (self.T <= 0), "Empty"
		
	def firm_state_transition(self,A):
		if self.action_space.check(A) == False:
			raise NameError('action out of bounds')
		return A
	
	def demand_transition(self):
		#self.demand = np.max([0.0,self.demand + 0.01*np.random.normal()])
		self.demand = np.min([100.0,np.max([0.0,self.demand + 0*np.random.choice([-1,0,1], p = [0.25,0.5,0.25])])])
		return self.demand
		
	def reset(self):
		# Should reset the state of all agents
		return self.S

class firm():
	'''
	This class models a generic firm and initializes its cost of production.
	The generic firm picks action at random.
	'''
	c = 0.0	#np.random.uniform()
	reward = None
	
	def __init__(self):
		self.reward = []
	
	def action(self):
		return np.random.uniform(0.0, q_max)
	
	def watch_market(self,game):
		pass

class static_nash_firm(firm):
	
	N = 1
	demand = q_max
		
	def watch_market(self,game):
		self.demand = game.demand
	
	def action(self):
		if self.demand > self.c:
			return (self.demand - self.c)/(self.N+1.0)
		else:
			return 0.0

class static_hetero_nash_firm(static_nash_firm):

	C_competitors = None
	
	def action(self):
		a = (self.demand/(self.N + 1.0)) - self.c + (self.C_competitors/(self.N+1.0))
		if a > 0.0:
			return a
		else:
			return 0.0

class static_BR_firm(static_nash_firm):
	
	Q_competitors = None

	def watch_market(self,game):
		self.demand = game.demand	
		self.Q_competitors = sum(game.action_history[-1,:])

	def action(self):
		if self.demand > (self.c + self.Q_competitors):
			return (self.demand - self.c - self.Q_competitors)/2.0
		else:
			return 0.0		

class deepRL_firm(firm):
	
	agent = None
	state = None
	
	def __init__(self,state_size,action_size):
		self.agent = D.DQNAgent(state_size, action_size)
		self.N = state_size
		self.reward = []
		
	def watch_market(self,game):
		self.state = state_vectorize(game.S)
	
	def action(self):
		return self.agent.get_action(self.state)

class RL_firm(firm):

	Q = None
	epsilon = 0.75
	epsilon_decay = 0.99999
	discount_factor=0.90
	alpha=0.5
	state = None

	def __init__(self,action_size):
		self.Q_value = defaultdict(lambda: np.zeros(action_size)) 
		self.reward = []
		self.policy = T.make_epsilon_greedy_policy(self.Q_value, action_size)
		
	def watch_market(self,game):
		self.state = state_tabularize(game.S)
		
	def action(self):
		self.epsilon = self.epsilon*self.epsilon_decay
		action_probs = self.policy(self.state, self.epsilon)
		return np.random.choice(np.arange(len(action_probs)), p=action_probs)
	
	def update_policy(self,state,action,reward,next_state):
            best_next_action = np.argmax(self.Q_value[next_state])    
            td_target = reward + self.discount_factor * self.Q_value[next_state][best_next_action]
            td_delta = td_target - self.Q_value[state][action]
            self.Q_value[state][action] += self.alpha * td_delta

def state_tabularize(game_state):
	''' 
	Input state is of the form [q1, q2, q3, ..., qN, d] 
	d is constant for a given game or episode
	qi is discrete with Q possible actions say Q = 10
	'''
	tab_state = 0
	for i in range(len(game_state)):
		tab_state += int(game_state[i])*np.power(Q,i)
	return tab_state
 
def state_vectorize(game_state):
	return np.reshape(game_state, [1, len(game_state)])
         
def play():
	players = [RL_firm(8), RL_firm(8),RL_firm(8),RL_firm(8)]
	N = len(players)
	demand_history = []
	action_history = []
	
	for e in range(EPISODES):
		done = False
		
		''' Initialize a Market. Players remain same across episodes. '''
		meeg = market_entry_exit_game(players)
		state = meeg.S
		
		while not done:
			''' Get each firms action '''
			a = np.zeros(N)
			for n in range(N):
				players[n].watch_market(meeg)
				a[n] = players[n].action()
			
			''' Interact with Market based on combined action of all firms '''
			next_state, reward, done, info = meeg.step(a)
			
			for n in range(N):
				players[n].reward.append(reward[n])
				
				''' Let firms update their policy '''
				if hasattr(players[n], 'agent'):
					players[n].agent.append_sample(state_vectorize(state), int(a[n]), reward[n], state_vectorize(next_state), done)
					players[n].agent.train_model()
				if hasattr(players[n], 'Q_value'):
					players[n].update_policy(state_tabularize(state),int(a[n]),reward[n],state_tabularize(next_state))
			
			state = next_state
			demand_history.append(meeg.demand)
			action_history.append(a)
			
			if done:
				for n in range(N):
					''' For DeepQ firms to update their target network'''
					if hasattr(players[n], 'agent'):              
						players[n].agent.update_target_model()
			
		print " Episode:" + str(e)
	
	visualize(players,demand_history,np.array(action_history))
	return players

def visualize(players,demand_history, action_history):
	plt.subplot(2,1,1)
	for p in players:
		plt.plot(range(len(demand_history) - 1 + 1),np.convolve(p.reward, np.ones((1,))/1, mode = 'valid'), lw = 0.1)
	plt.plot(range(len(demand_history)),map(lambda x: x*x/4.0, demand_history), lw = 1.0)
	plt.subplot(2,1,2)
	for p in range(len(players)):
		plt.plot(range(len(action_history) - 1 + 1),action_history[:,p], lw = 0.1)
	plt.plot(range(len(demand_history)),map(lambda x: x/2.0, demand_history), lw = 1.0)
	plt.show()
	pass
