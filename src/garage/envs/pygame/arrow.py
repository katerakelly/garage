import pygame
import numpy as np
from ple.games import Catcher
from ple.games.catcher import Fruit
from ple.games.catcher import Paddle


class ArrowFruit(Fruit):

    def __init__(self, delay, speed, size, *args, **kwargs):
        super().__init__(speed, size, *args, **kwargs)

        # delay controls time steps displaying arrow
        self.delay = delay
        self.timestep = 0

        arrow_image = pygame.Surface((size, size))
        arrow_image.fill((0, 0, 0, 0))
        arrow_image.set_colorkey((0,0,0))

        pygame.draw.rect(
                arrow_image,
                (255, 255, 255),
                (0, 0, size / 2, size),
                0
        )

        self.arrow_image = arrow_image
        self.arrow_rect = self.arrow_image.get_rect()
        self.arrow_rect.center = (-30, -30)

    def update(self, dt):
        if self.timestep < self.delay:
            # don't drop the fruit yet
            self.timestep += 1
        else:
            super().update(dt)

    def reset(self):
        # NOTE removed *2 here to allow more x-range
        x = self.rng.choice( range(self.size, self.SCREEN_WIDTH-self.size, self.size) )

        # NOTE very last step of episode will have the arrow instead of the fruit...
        self.rect.center = (x, 0)
        self.arrow_rect.center = (x, 0)
        self.timestep = 0

    def draw(self, screen):
        if self.timestep < self.delay:
            screen.blit(self.arrow_image, self.arrow_rect.center)
        else:
            screen.blit(self.image, self.rect.center)

    def get_state(self):
        if self.timestep < self.delay:
            # return state of the arrow
            return self.arrow_rect.center[0], self.arrow_rect.center[1]
            # TODO remove
            #return 0, 0
        else:
            # return state of fruit
            return self.rect.center[0], self.rect.center[1]


class Arrow(Catcher):
    '''
    Modify catcher game to make it impossible to catch
    the fruit unless you follow the direction of an arrow
    that appears at the top of the screen where the fruit
    will fall
    '''
    def __init__(self, *args, delay=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.fruit_fall_speed = 4 * self.fruit_fall_speed
        self.player_speed = 0.25 * self.player_speed
        print('Setting delay in arrow to: {}'.format(delay))
        self.delay = delay

    def init(self):
        self.score = 0
        self.lives = self.init_lives

        self.player = Paddle(self.player_speed, self.paddle_width,
                self.paddle_height, self.width, self.height)

        self.fruit = ArrowFruit(self.delay, self.fruit_fall_speed, self.fruit_size,
                self.width, self.height, self.rng)

        self.fruit.reset()

    def getGameState(self):
        """
        add position of arrow to state
        """
        fruit_x, fruit_y = self.fruit.get_state()
        state = {
            "player_x": self.player.rect.center[0],
            "player_vel": self.player.vel,
            "fruit_x": fruit_x,
            "fruit_y": fruit_y,
        }

        return state

    def distance2fruit(self):
        """
        return x-distance from agent to fruit
        """
        if self.fruit.timestep < self.fruit.delay:
            # fruit is not visible, just return max distance
            return 36
        else:
            fruit_x, _ = self.fruit.get_state()
            player_x = self.player.rect.center[0]
            return np.abs(player_x - fruit_x)
