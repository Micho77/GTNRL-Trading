from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque
from keras.layers import Dropout


class Agent:

    def __init__(self, state_size, balance, is_eval=False, model_name="", model_target_name=""):

        # Training / agent configurations
        self.state_size = state_size 
        self.action_size = 2 # buy, sell
        self.memory = deque(maxlen=100000)
        self.buy_inventory = []
        self.sell_inventory = []
        self.balance = balance
        self.model_name = model_name
        self.model_target_name = model_target_name
        self.is_eval = is_eval
        self.gamma = 0.99 
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.85

        # Model configurations
        self.model = load_model("models/" + model_name) if is_eval else self.model()
        self.target_model = load_model("models/"+ model_target_name) if is_eval else self.model

    def model(self):

        model = Sequential()
        model.add(Dense(units=256, input_dim=self.state_size, activation="relu"))
        # # model.add(Dropout(0.5))
        # model.add(Dense(units=256, activation="relu"))
        # # model.add(Dropout(0.5))
        # model.add(Dense(units=256, activation="relu"))
        # # model.add(Dropout(0.5))
        # model.add(Dense(units=256, activation="relu"))
        # # model.add(Dropout(0.5))
        # model.add(Dense(units=256, activation="relu"))
        # # model.add(Dropout(0.5))
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
        minibatch = random.sample(self.memory, batch_size)
        stateBatch = np.zeros((batch_size, self.state_size))
        targetBatch = np.zeros((batch_size, self.action_size))
        x = 0
        for state, action, reward, next_state, done in minibatch:
            stateBatch[x] = state[0]
            targetBatch[x] = self.model.predict(state)[0]
            t = self.target_model.predict(next_state)[0]
            targetBatch[x][action] = reward + (1-done)*self.gamma * np.amax(t)
            x+=1
        self.model.fit(stateBatch, targetBatch, epochs=1, verbose=0)
