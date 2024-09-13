import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from snake_game import SnakeGame, ACTIONS
import random
from collections import deque
import matplotlib.pyplot as plt



# Hiperparametreler
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
MEMORY_SIZE = 16000
BATCH_SIZE = 2
EPSILON = 1.0
EPSILON_DECAY = 0.99  # Epsilon hızlı bir şekilde azalacak
MIN_EPSILON = 0.01
DYNAMIC_LEARNING_RATE = True  # Dinamik öğrenme oranı

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.learning_rate = LEARNING_RATE
        self.model = self.build_model()

    def build_model(self):

        model = models.Sequential()
        model.add(layers.Dense(8, input_dim=self.state_size, activation='relu'))  # Daha küçük katman
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(ACTIONS)  # Rastgele hareket (keşif)
        q_values = self.model.predict(state)
        return ACTIONS[np.argmax(q_values[0])]  # En iyi aksiyon (sömürü)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + DISCOUNT_FACTOR * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            action_idx = ACTIONS.index(action)
            target_f[0][action_idx] = target

            if episode % 10 == 0:
                self.model.fit(state, target_f, epochs=1, verbose=0)


        if DYNAMIC_LEARNING_RATE:
            self.learning_rate = max(LEARNING_RATE * EPSILON_DECAY, 0.0001)
            self.model.optimizer.learning_rate = self.learning_rate

        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    game = SnakeGame()
    state_size = 4  # [snake_x, snake_y, food_x, food_y]
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)


    scores = []
    total_rewards = []

    for episode in range(500):
        state = game.get_state().reshape(1, -1)
        game_over = False
        total_reward = 0

        while not game_over:
            action = agent.act(state)
            next_state, reward, game_over, score = game.step(action)
            next_state = next_state.reshape(1, -1)
            agent.remember(state, action, reward, next_state, game_over)
            state = next_state
            total_reward += reward

            game.render()

            if game_over:
                print(f"Episode: {episode}, Score: {score}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, LR: {agent.learning_rate:.6f}")
                scores.append(score)
                total_rewards.append(total_reward)
                game.reset()


        agent.replay()

        if episode % 50 == 0:
            agent.save(f"snake_dqn_weights_{episode}.h5")


    plt.figure(figsize=(12, 5))

    # Skor grafiği
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title("Episode vs Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    # Ödül grafiği
    plt.subplot(1, 2, 2)
    plt.plot(total_rewards)
    plt.title("Episode vs Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.tight_layout()
    plt.show()
