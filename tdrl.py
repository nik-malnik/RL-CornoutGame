import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import games as G

from collections import defaultdict

def make_epsilon_greedy_policy(Q, nA):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
    
def td_q(discount_factor=1.0, alpha=0.5, epsilon=0.1):
    
    players = [G.firm()]
    meeg = G.market_entry_exit_game(players)
    state_size = (meeg.N + 1)
    action_size = 10
    EPISODES = 20
    
    stats = {'episode_num': [], 'episode_reward': []}

    Q = defaultdict(lambda: np.zeros(action_size)) 

    policy = make_epsilon_greedy_policy(Q, epsilon, action_size)
    
    for e in range(EPISODES):
        
        meeg = G.market_entry_exit_game(players)
        state = int(meeg.S[-1])
		
        episode_reward = 0.0
        for t in itertools.count():

            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = meeg.step(action)
            next_state = int(next_state[-1])
            
            episode_reward += reward[0]
            
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
        
        stats['episode_num'].append(e)
        stats['episode_reward'].append(episode_reward)
    
    return Q,stats

def td_sarsa(discount_factor=1.0, alpha=0.5, epsilon=0.1):
    
    players = [G.firm()]
    meeg = G.market_entry_exit_game(players)
    state_size = (meeg.N + 1)
    action_size = 10
    EPISODES = 20
    
    stats = {'episode_num': [], 'episode_reward': []}

    Q = defaultdict(lambda: np.zeros(action_size)) 

    policy = make_epsilon_greedy_policy(Q, epsilon, action_size)
    
    for e in range(EPISODES):
        
        meeg = G.market_entry_exit_game(players)
        state = int(meeg.S[-1])
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
		
        episode_reward = 0.0
        for t in itertools.count():

            next_state, reward, done, _ = meeg.step(action)
            next_state = int(next_state[-1])

            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            episode_reward += reward[0]
            
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            action = next_action    
            state = next_state
        
        stats['episode_num'].append(e)
        stats['episode_reward'].append(episode_reward)
    
    return Q,stats
