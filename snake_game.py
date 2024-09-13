import pygame
import random
import numpy as np

# Oyun ayarları
WIDTH, HEIGHT = 500, 500
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

ACTIONS = [UP, DOWN, LEFT, RIGHT]

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice(ACTIONS)
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.previous_distance = self.calculate_distance(self.snake[0], self.food)

    def generate_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def move_snake(self):
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        self.snake.insert(0, new_head)

        if self.snake[0] == self.food:
            self.score += 1
            self.food = self.generate_food()
        else:
            self.snake.pop()

    def is_collision(self):
        head = self.snake[0]
        return (
            head[0] < 0 or head[0] >= GRID_WIDTH or
            head[1] < 0 or head[1] >= GRID_HEIGHT or
            head in self.snake[1:]
        )

    def calculate_distance(self, point1, point2):
        # Manhattan mesafesi hesaplama
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def step(self, action):
        self.direction = action
        self.move_snake()

        # Yılanın önceki ve şimdiki yemeğe uzaklığını hesapla
        current_distance = self.calculate_distance(self.snake[0], self.food)
        reward = 0

        if self.is_collision():
            reward = -200
            self.game_over = True
        elif self.snake[0] == self.food:
            reward = 300
            self.previous_distance = self.calculate_distance(self.snake[0], self.food)  # Yemeği yedikten sonra sıfırla
        else:

            if current_distance < self.previous_distance:
                reward = 10  # Yaklaştığı için pozitif ödül
            else:
                reward = 0  # Uzaklaştığı için negatif ödül
            self.previous_distance = current_distance

        return self.get_state(), reward, self.game_over, self.score

    def get_state(self):
        head = self.snake[0]
        return np.array([head[0], head[1], self.food[0], self.food[1]])

    def render(self):
        self.screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, RED, (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.display.flip()

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    game = SnakeGame()
    while True:
        game.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                sys.exit()
