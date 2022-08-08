# This Python file aims at implementing a the Advantage Actor-Critic (A2C) approach to solve the CartPole-v0 task of the OpenAI Gym Environment

# Importing all the required frameworks
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model
import keras.backend as k
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from envtest import envtest
import random


# A function to keep track of average rewards
def running_reward_avg(rewards):
    output = []
    for i in range(len(rewards)):
        output.append(sum(rewards[:i]) / (i + 1))
    return output


# Initializing the Agent class which controls the agents behaviour/improvement in the environment
class Agent(object):

    def __init__(self, env, gamma=0.98, alpha=0.01):

        self.env = env
        self.state_size = 5  # Number of features describing each state in the environment
        self.action_size = 1  # Number of features describing each action in the environment
        self.gamma = gamma  # Discount factor for future rewards
        self.alpha = alpha  # Learning rate during training
        self.n_hl1 = 16  # Number of units in the first hidden layer of the network
        self.n_hl2 = 16  # Number of units in the second hidden layer of the network
        self.actor, self.critic, self.policy = self.build_network()  # Building the networks that output different output layers
        self.reward_history = []  # Reward history to keep track of rewards per episode
        self.episode_lengths = []  # To keep track of the length of each episode

    # Defining a function that builds the actor, critic, and policy network to train/improve the model
    def build_network(self):

        inputs = Input(shape=[self.state_size])
        advantage = Input(shape=[1])
        X = Dense(self.n_hl1, activation='relu')(inputs)
        X = Dense(self.n_hl2, activation='relu')(X)
        actor_layer = Dense(self.action_size, activation='softmax')(X)
        critic_layer = Dense(1)(X)

        # Defining a custom loss function (in the keras format) for the policy gradient loss
        def custom_loss(y_pred, y_true):
            probs = k.clip(y_pred, 1e-10, 1 - 1e-10)
            return -k.mean(k.log(probs) * y_true) * advantage

        # The policy network takes the state and reward obtained, to calculate probabilities for each action(stochastic policy)
        actor_model = Model(inputs=[inputs, advantage], outputs=actor_layer)
        actor_model.compile(optimizer=Adam(learning_rate=self.alpha), loss=custom_loss)

        # The critic network takes the state, and calculates the value of that state
        critic_model = Model(inputs=inputs, outputs=critic_layer)
        critic_model.compile(optimizer=Adam(learning_rate=self.alpha), loss='mean_squared_error')

        # The policy network is like the actor network, but doesnt take any reward, and just predicts the stochastic probabilities
        # Training is not done on this network, but on the actor network instead
        policy_model = Model(inputs=inputs, outputs=actor_layer)

        return actor_model, critic_model, policy_model

    # Choosing a greedy action, except certain times randomly (randomness can be modified with changing epsilon)
    def choose_action(self, state, epsilon=0.2):

        policy = self.policy.predict(state.reshape([1, self.state_size]))
        if np.random.rand(1) < epsilon:
            return np.argmax(policy[0])
        else:
            return np.random.randint(self.action_size)

    # Updating the actor-critic networks after every timestep in an episode
    def update_network(self, state_now, action, reward, state_next):

        state_now = state_now.reshape([1, self.state_size])
        state_next = state_next.reshape([1, self.state_size])
        action = (np.eye(self.action_size)[action]).reshape([1, self.action_size])
        state_now_value = self.critic.predict(state_now)
        state_next_value = self.critic.predict(state_next)
        target_value_now = reward + self.gamma * state_next_value
        advantage = target_value_now - state_now_value
        self.actor.fit([state_now, advantage], action)
        self.critic.fit(state_now, target_value_now)

    # Training the agent over a number of episodes, so it learns the policy to claim maximum reward
    def train(self, num_episodes=5000):

        ep_pointer = 0
        # Iterating over 'num_episodes'
        for i in range(num_episodes):

            # Maintaining a buffer to store the reward obtained throughout each episode
            reward_buffer = 0
            # Maintaining the count of the length of the episode
            j = 0
            # Resetting the starting state of each episode
            state_now, ep_pointer = env.reset(ep_pointer)
            ep_length = 0
            while True:
                # Choosing an action as per policy, and going to the next state, claiming reward
                action = agent.choose_action(state_now)
                state_next, reward, done, _ = env.step(action, ep_length)
                # Incrementing the reward buffer, and step count of the episode
                reward_buffer += reward
                j += 1
                # Updating the networks
                agent.update_network(state_now, action, reward, state_next)
                # Seeing if the episode terminates
                if done == True:
                    self.reward_history.append(reward_buffer)
                    self.episode_lengths.append(j)
                    if (i + 1) % 100 == 0:
                        print("Length of Episode {} : {}".format(i + 1, self.episode_lengths[-1]))
                        print("Total reward claimed by the agent in episode {} : {}".format(i + 1,
                                                                                            self.reward_history[-1]))
                    break
                else:
                    # Else setting the next timestep's state
                    state_now = state_next
                ep_length += 1
            ep_pointer += max(ep_length, 1)


# Creating an environment, and an agent

discreate_road_pd = pd.read_excel('sin road.xlsx')
road = discreate_road_pd.to_numpy()
env = envtest(road)

agent = Agent(env)

# Training the agent using Deep Q-Learning with experience replay with the above mentioned parameters
agent.train()

# Plotting the results
fig, axs = plt.subplots(1, 2)
axs[0].plot(running_reward_avg(agent.reward_history))
axs[0].set_title('Average Reward per Episode')
axs[1].plot(agent.episode_lengths, 'tab:orange')
axs[1].set_title('Episode_Length')

plt.show()
