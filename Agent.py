from GenericNetwork import GenericNetwork
from keras.optimizers import adam_v2
import tensorflow_probability as tfp
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, layer1_dim=128, layer2_dim=64, n_actions=2, alpha_A=0.00003, alpha_C=0.00005, gamma=0.99):
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha_A = alpha_A
        self.alpha_C = alpha_C
        self.action = None
        self.log_prob = None

        self.actor = GenericNetwork(n_actions, layer1_dim, layer2_dim, "actor")
        self.actor.compile(optimizer=adam_v2.Adam(learning_rate=alpha_A))
        self.critic = GenericNetwork(1, layer1_dim, layer2_dim, "critic")
        self.critic.compile(optimizer=adam_v2.Adam(learning_rate=alpha_C))
        self.aloss = []
        self.closs = []

    def choose_action(self, observation):  # obs shape (1,2)
        state = tf.convert_to_tensor([observation])  # state shape (1,1,2)
        pars = self.actor(state)  # mean and standard deviation that make action probs
        pars = np.asarray(tf.squeeze(pars)).reshape(1, 2)
        sigma, mu = np.hsplit(pars, 2)
        sigma = tf.exp(sigma)  # get rid of negative sigma
        # sigma= abs(sigma)
        action_probabilities = tfp.distributions.Normal(mu, sigma)  # normal distribution with mu,sigma pars
        # log_prob = action_probabilities.log_prob(action_probabilities) #log (gonna be used for gradient)
        action = action_probabilities.sample()  # choose action (most likely to be chosen with higher probability)
        action = tf.tanh(action) * 0.07  # action: continuous num in range(-0.07, 0.07)((-4,4) degree_
        self.action = action
        return action  # cast tensor to numpy(openAI gym doesnt take tensor)

    # def save_models(self):
    #     #print('... saving models ...')
    #     self.actor.save_weights(self.actor.checkpoint_file)
    #     self.critic.save_weights(self.critic.checkpoint_file)
    # def load_models(self):
    #     print('... loading models ...')
    #     self.actor.load_weights(self.actor.checkpoint_file)
    #     self.critic.load_weights(self.critic.checkpoint_file)

    def learn(self, state, reward, state_, done):
        # print("state before ")
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)  # not fed to NN -> no need to reshape
        with tf.GradientTape(persistent=True) as tape:
            state_value = self.critic(state)
            state_value_ = self.critic(state_)
            state_value = tf.squeeze(state_value)  # squeeze Removes dims of size 1 from the shape of a tensor.
            state_value_ = tf.squeeze(state_value_)
            pars = self.actor(state)
            # pars= np.asarray(tf.squeeze(pars)).reshape(1,2)
            # mu , sigma= np.hsplit(pars , 2)
            # mu = np.squeeze(mu)
            # sigma = np.squeeze(sigma)
            mu = pars[0, 0]
            sigma = pars[0, 1]
            # print(sigma)
            # sigma = tf.exp(sigma)
            # print(sigma)
            action_probs = tfp.distributions.Normal(mu, abs(sigma))  # policy
            log_prob = action_probs.log_prob(self.action[0, 0])
            # print(mu,sigma)
            # print(log_prob)

            # TD error:
            TD = self.gamma * state_value_ * (1 - int(done)) - state_value
            delta = reward + TD  # 1-done: terminal stRemoves dimensions of size 1 from the shape of a tensor.ate zero effect
            actor_loss = (-log_prob * delta)
            critic_loss = (delta ** 2)
            # print("sig", sigma , "ac", actor_loss, "cr", critic_loss)

        gradient1 = tape.gradient(actor_loss, self.actor.trainable_variables)

        self.actor.optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(gradient1, self.actor.trainable_variables))
        # if grad is not None

        gradient2 = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            (grad, var) for (grad, var) in zip(gradient2, self.critic.trainable_variables))
        # if grad is not None
        return critic_loss, actor_loss, gradient1