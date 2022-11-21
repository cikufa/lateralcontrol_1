import tensorflow as tf
from Agenttest import Agent
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow_probability as tfp

model = keras.models.load_model('model/model500')
loc= np.array([0.0]).reshape((1, 1)) 
sim = 20
j=0
a=0
for i in range(sim):
    state = tf.convert_to_tensor([loc], dtype=tf.float32) 
    # print(state)
    k = model.call(state)
    pars = np.asarray(tf.squeeze(k)).reshape(1, 2)
    mu , sigma = np.hsplit(pars, 2)
    sigma = abs(sigma)  
    action_probabilities = tfp.distributions.Normal(mu, sigma) 
    action = action_probabilities.sample() 
    action = tf.tanh(action)  # a
    a = a +action
    print(action)
    loctmp= loc 
    loc = loc + action 
    xs = [loctmp[0], loc[0]]; ys=[j , j+0.3]
    j= j+0.3
    plt.plot(xs ,ys , 'bo', linestyle= '--' )
print(a/sim) 
plt.show()
# plt.savefig('simulation.png')    
        

# loc= np.array([0]).reshape((1, 1)) 
# for i in range(10):
#     state = tf.convert_to_tensor([loc], dtype=tf.float32)
#     pars = ag.actor(state)
#     print(pars)

# d =np.loadtxt('data.csv')
# ind= np.arange(0,d.shape[0])

# plt.plot(ind, d)
# plt.show()