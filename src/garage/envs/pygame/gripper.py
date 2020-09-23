from ple.games import Catcher
from ple.games.catcher import Paddle
import pygame


class ColoredPaddle(Paddle):

    def __init__(self, speed, width, height, SCREEN_WIDTH, SCREEN_HEIGHT):
        self.speed = speed
        self.width = width

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.vel = 0.0

        pygame.sprite.Sprite.__init__(self)

        image = pygame.Surface((width, height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0,0,0))

        pygame.draw.rect(
                image,
                (0, 255, 0), #NOTE make it green!
                (0, 0, width, height),
                0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH/2 - self.width/2, SCREEN_HEIGHT-height-3)


class Gripper(Catcher):
    """
    Modify classic catcher game to make the agent a gripper rather than a paddle
    An additional action controls whether the gripper is open or closed
    The gripper must be open to catch fruit

    The state of the gripper is indicated by the color of the agent
    white -> gripper is closed
    green -> gripper is open

    This scheme is implemented by having two players, which are drawn with
    different colors
    A different player is active based on the state of the gripper
    Both players are updated at each step, but only the active player
    is visualized
    """

    def init(self):
        super().init()
        # add new action to open/close gripper
        open_action = {'open': 0}
        self.actions.update(open_action)
        # implement w/ two Paddle objects, one for open, one for closed
        self.player_open = ColoredPaddle(self.player_speed, self.paddle_width,
                self.paddle_height, self.width, self.height)
        self.active_player = self.player_open # initialize gripper closed

    def _handle_player_events(self):
        self.dx = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions['left']:
                    self.dx -= self.player_speed

                if key == self.actions['right']:
                    self.dx += self.player_speed

                # if opn/close action is taken, switch the active player
                if key == self.actions['open']:
                    if self.active_player == self.player:
                        self.active_player = self.player_open
                    else:
                        self.active_player = self.player

    def getGameState(self):
        """
        return state of active player
        add gripper open/close to state
        """
        state = {
            "player_x": self.active_player.rect.center[0],
            "player_vel": self.active_player.vel,
            "fruit_x": self.fruit.rect.center[0],
            "fruit_y": self.fruit.rect.center[1],
            "gripper": 0 if self.active_player == self.player else 1
        }

        return state

    def step(self, dt):
        """
        step both players, draw the active one
        receive positive reward if fruit is caught with
        the open gripper, negative reward if closed gripper
        """
        self.screen.fill((0,0,0))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        if self.fruit.rect.center[1] >= self.height:
            self.score += self.rewards["negative"]
            self.lives -= 1
            self.fruit.reset()

        # get reward only if fruit caught with open gripper
        if pygame.sprite.collide_rect(self.player, self.fruit):
            if self.active_player == self.player_open:
                self.score += self.rewards["positive"]
            else:
                self.score += self.rewards["negative"]
            self.fruit.reset()

        # keep both players updated so in the same place
        self.player.update(self.dx, dt)
        self.player_open.update(self.dx, dt)
        self.fruit.update(dt)

        if self.lives == 0:
            self.score += self.rewards["loss"]

        # draw the player that is currently active
        self.active_player.draw(self.screen)
        self.fruit.draw(self.screen)
