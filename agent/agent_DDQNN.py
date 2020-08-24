from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque


class Agent:

    def __init__(self, state_size=None, balance=0, is_eval=False, model_name="", model_target_name=""):

        # Training / agent configurations
        self.state_size = state_size 
        self.action_size = 2  # buy, sell
        self.memory = deque(maxlen=100000)
        self.model_name = model_name
        self.model_target_name = model_target_name
        self.is_eval = is_eval
        self.gamma = 0.99 
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.85
        self.episode_tot_reward = 0
        self.episode_rewards = []

        # Model configurations
        self.model = load_model("models/" + model_name) if is_eval else self.model()
        self.target_model = load_model("models/"+ model_target_name) if is_eval else self.model

    def episode_reset(self):

        # Reset episode variables
        self.episode_tot_reward = 0
        self.episode_rewards = []

    def model(self):

        model = Sequential()
        model.add(GRU(units=32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def replay(self, batch_size):

        # sub-sampling memory
        minibatch = random.sample(self.memory, batch_size)

        # TODO: STORE SOME OF THESE DIRECTLY IN MEMORY
        # Get vectors and matrices from minibatch
        states, actions, rewards, next_states, dones = zip(*minibatch)
        stateBatch = np.array(states)
        nextStateBatch = np.array(next_states)
        targetBatch = self.model.predict(stateBatch)

        # Update values using target model
        new_vals = np.array(rewards) + self.gamma*self.target_model.predict(nextStateBatch).max(1)
        list_idxs = (list(range(targetBatch.shape[0])), actions)
        targetBatch[list_idxs] = new_vals

        self.model.fit(stateBatch, targetBatch, epochs=1, verbose=0)
