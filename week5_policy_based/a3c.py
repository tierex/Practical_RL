from __future__ import print_function, division
from IPython.core import display
import matplotlib.pyplot as plt
import numpy as np

#If you are running on a server, launch xvfb to record game videos
#Please make sure you have xvfb installed
import os

import gym
from atari_util import PreprocessAtari

def make_env():
    env = gym.make("KungFuMasterDeterministic-v0")
    env = PreprocessAtari(env, height=42, width=42,
                          crop = lambda img: img[60:-30, 5:],
                          dim_order = 'tensorflow',
                          color=False, n_frames=4,
                          reward_scale = 0.01)
    return env

env = make_env()

obs_shape = env.observation_space.shape
n_actions = env.action_space.n

print("Observation shape:", obs_shape)
print("Num actions:", n_actions)
print("Action names:", env.env.env.get_action_meanings())


s = env.reset()
for _ in range(100):
    s, _, _, _ = env.step(env.action_space.sample())


import tensorflow as tf
tf.reset_default_graph()
sess = tf.InteractiveSession()

from keras.layers import Conv2D, Dense, Flatten, Input
import keras
from keras.models import Model, Sequential
import tensorflow as tf


class Agent:
    def __init__(self, name, state_shape, n_actions, reuse=False):
        """A simple actor-critic agent"""

        with tf.variable_scope(name, reuse=reuse):
            # Prepare neural network architecture
            ### Your code here: prepare any necessary layers, variables, etc.

            # prepare a graph for agent step
            inputs = Input(shape=state_shape)
            body = Conv2D(32, (3, 3), strides=2, activation='relu')(inputs)
            body = Conv2D(32, (3, 3), strides=2, activation='relu')(body)
            body = Conv2D(32, (3, 3), strides=2, activation='relu')(body)
            body = Flatten()(body)
            body = Dense(128, activation='relu')(body)

            logits = Dense(n_actions, activation='linear')(body)
            state_values = Dense(1, activation='linear')(body)

            self.network = Model(inputs=inputs, outputs=[logits, state_values])

            self.state_t = tf.placeholder('float32', [None, ] + list(state_shape))
            self.agent_outputs = self.symbolic_step(self.state_t)

    def symbolic_step(self, state_t):
        """Takes agent's previous step and observation, returns next state and whatever it needs to learn (tf tensors)"""

        # Apply neural network
        ### Your code here: apply agent's neural network to get policy logits and state values.

        logits, state_value = self.network(state_t)

        state_value = state_value[:, 0]
        assert tf.is_numeric_tensor(state_value) and state_value.shape.ndims == 1, \
            "please return 1D tf tensor of state values [you got %s]" % repr(state_value)
        assert tf.is_numeric_tensor(logits) and logits.shape.ndims == 2, \
            "please return 2d tf tensor of logits [you got %s]" % repr(logits)
        # hint: if you triggered state_values assert with your shape being [None, 1],
        # just select [:, 0]-th element of state values as new state values

        return (logits, state_value)

    def step(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tf.get_default_session()
        return sess.run(self.agent_outputs, {self.state_t: state_t})

    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_values = agent_outputs
        policy = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        return np.array([np.random.choice(len(p), p=p) for p in policy])


agent = Agent("agent", obs_shape, n_actions)
sess.run(tf.global_variables_initializer())

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

state = [env.reset()]
logits, value = agent.step(state)
print("action logits:\n", logits)
print("state values:\n", value)


class EnvBatch:
    def __init__(self, n_envs=10):
        """ Creates n_envs environments and babysits them for ya' """
        self.envs = [make_env() for _ in range(n_envs)]

    def reset(self):
        """ Reset all games and return [n_envs, *obs_shape] observations """
        return np.array([env.reset() for env in self.envs])

    def step(self, actions):
        """
        Send a vector[batch_size] of actions into respective environments
        :returns: observations[n_envs, *obs_shape], rewards[n_envs], done[n_envs,], info[n_envs]
        """
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        new_obs, rewards, done, infos = map(np.array, zip(*results))

        # reset environments automatically
        for i in range(len(self.envs)):
            if done[i]:
                new_obs[i] = self.envs[i].reset()

        return new_obs, rewards, done, infos


def evaluate(agent, env, n_games=1):
    """Plays an a game from start till done, returns per-game rewards """

    game_rewards = []
    for _ in range(n_games):
        state = env.reset()

        total_reward = 0
        while True:
            action = agent.sample_actions(agent.step([state]))[0]
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done: break

        game_rewards.append(total_reward)
    return game_rewards


env_monitor = gym.wrappers.Monitor(env, directory="sample_kungfu_videos", force=True)
rw = evaluate(agent, env_monitor, n_games=3,)
env_monitor.close()
print (rw)