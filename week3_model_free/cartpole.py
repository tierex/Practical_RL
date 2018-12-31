import numpy as np
import matplotlib.pyplot as plt
import gym

from qlearning import QLearningAgent

import random


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = []
        self._maxsize = size
        self._currsize = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        '''
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''
        data = (obs_t, action, reward, obs_tp1, done)

        self._storage.append(data)
        if len(self._storage) > self._maxsize:
            self._storage.pop(0)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = np.random.choice(self.__len__(), size=min(batch_size, self.__len__()), replace=False)

        # collect <s,a,r,s',done> for each index
        samples = list(self._storage[ind] for ind in idxes)
        states, actions, rewards, next_states, is_done = zip(*samples)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(is_done)


class EVSarsaAgent(QLearningAgent):
    """
    An agent that changes some of q-learning functions to implement Expected Value SARSA.
    Note: this demo assumes that your implementation of QLearningAgent.update uses get_value(next_state).
    If it doesn't, please add
        def update(self, state, action, reward, next_state):
            and implement it for Expected Value SARSA's V(s')
    """

    def get_value(self, state):
        """
        Returns Vpi for current state under epsilon-greedy policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        Hint: all other methods from QLearningAgent are still accessible.
        """
        epsilon = self.epsilon
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        best_action = self.get_best_action(state)
        q_best_action = self.get_qvalue(state, best_action)

        state_value = (1 - epsilon) * q_best_action

        state_value += sum(
            [epsilon / len(possible_actions) * self.get_qvalue(state, action) for action in possible_actions])

        return state_value


def play_and_train(env, agent, t_max=10 ** 4, replay=None, replay_batch_size=32):
    """This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward"""
    total_reward = 0.0
    s = env.reset()
    
    for t in range(t_max):
        a = agent.get_action(s)

        next_s, r, done, _ = env.step(a)
        agent.update(s, a, r, next_s)
        if replay is not None:
            # store current <s,a,r,s'> transition in buffer
            replay.add(s, a, r, next_s, done)

            # sample replay_batch_size random transitions from replay,
            # then update agent on each of them in a loop
            rep_s_arr, rep_a_arr, rep_r_arr, rep_next_s_arr, _ = replay.sample(replay_batch_size)
            for i in range(len(rep_s_arr)):
                agent.update(tuple(rep_s_arr[i]), rep_a_arr[i], rep_r_arr[i], tuple(rep_next_s_arr[i]))

        s = next_s
        total_reward += r
        if done: break

    return total_reward

def obs_to_state(env, obs):
    """ Maps an observation to state """
    n_states = 80
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b


from gym.core import ObservationWrapper
class Binarizer(ObservationWrapper):

    def _observation(self, state):
        # state = <round state to some amount digits.>
        # hint: you can do that with round(x,n_digits)
        # you will need to pick a different n_digits for each dimension

        state[0] = np.round(state[0], 0)
        state[1] = np.round(state[1], 1)
        state[2] = np.round(state[2], 2)
        state[3] = np.round(state[3], 1)
        return tuple(state)

class MountainCarBinarizer(ObservationWrapper):

    def _observation(self, state):
        # state = <round state to some amount digits.>
        # hint: you can do that with round(x,n_digits)
        # you will need to pick a different n_digits for each dimension

        # state[0] = np.round(state[0], 1)
        # state[1] = np.round(state[1], 2)
        return obs_to_state(self, state)

def play(env, agent):
    total_reward = 0.0
    s = env.reset()

    while True:
        env.render()
        a = agent.get_best_action(s)
        next_s, r, done, _ = env.step(a)
        s = next_s
        total_reward += r
        if done:
            total_reward = 0.0
            s = env.reset()

from qlearning import QLearningAgent

def train_ad_play_agents(env, agent, max_iter = 10**4, replay = None):
    rewards = []
    for i in range(max_iter):
        agent.epsilon *= 0.99
        rewards.append(play_and_train(env, agent, replay=replay))
        # Note: agent.epsilon stays constant
        if i % 100 == 0:
            print('mean reward =', np.mean(rewards[-100:]), 'epsilon',agent.epsilon)
    agent.epsilon = 0.0
    play(env, agent)


import gym, gym.envs.toy_text
# env = Binarizer(gym.make("CartPole-v0"))
env = MountainCarBinarizer(gym.make("MountainCar-v0"))

replay = ReplayBuffer(1000)
agent_sarsa = EVSarsaAgent(alpha=0.25, epsilon=0.3, discount=0.99,
                           get_legal_actions=lambda s: range(n_actions))

agent_ql = QLearningAgent(alpha=0.25, epsilon=0.3, discount=1.0,
                          get_legal_actions=lambda s: range(n_actions))
n_actions = env.action_space.n

train_ad_play_agents(env, agent_sarsa, max_iter=10000, replay=replay)