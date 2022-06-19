from keras.layers import Dense
import os
from tensorflow import keras

class GenericNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims, name, chkpt_dir="/tmp/actor_critic"):
        super(GenericNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions)

        # self.v = Dense(1, activation=None)
        # continous action is represented as a normal distribution that is characterized with 2 quantities: a mean and a standard deviation
        # self.pi = Dense(n_actions=2, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return x