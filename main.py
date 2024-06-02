import math
import random
import sys
import pygame
import neat

WINDOW_WIDTH = 288
WINDOW_HEIGHT = 512

BIRD_WIDTH = 34
BIRD_HEIGHT = 24
BIRD_TEXTURE1 = 'assets/yellowbird-downflap.png'
BIRD_TEXTURE2 = 'assets/yellowbird-midflap.png'
BIRD_TEXTURE3 = 'assets/yellowbird-upflap.png'

PIPE_TEXTURE = 'assets/pipe-green.png'
PIPE_WIDTH = 52
PIPE_HEIGHT = 320

BG_TEXTURE = 'assets/background-day.png'

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

clock = pygame.time.Clock()
font = pygame.font.SysFont("lucidasans", 24)

GENERATIONS = 50
CONFIG_FILE = 'config.txt'
CONFIG = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                            neat.DefaultStagnation, CONFIG_FILE)
bg = pygame.transform.scale(pygame.image.load(BG_TEXTURE), (WINDOW_WIDTH, WINDOW_HEIGHT))
screen.blit(bg, (0, 0))

speed = 10


class Pipe:
    def __init__(self):
        self.x = WINDOW_WIDTH
        self.y = random.randint(WINDOW_HEIGHT - PIPE_HEIGHT + 50, WINDOW_HEIGHT)
        self.texture = pygame.transform.scale(pygame.image.load(PIPE_TEXTURE).convert(), (PIPE_WIDTH, PIPE_HEIGHT))
        self.inverted_texture = pygame.transform.scale(pygame.image.load(PIPE_TEXTURE).convert(),
                                                       (PIPE_WIDTH, PIPE_HEIGHT))
        self.inverted_texture = pygame.transform.flip(self.inverted_texture, False, True)

    def update(self):
        self.x -= speed

    def draw(self):
        screen.blit(self.texture, (self.x, self.y))
        screen.blit(self.inverted_texture, (self.x, self.y - WINDOW_HEIGHT))


class Bird:
    def __init__(self, nn):
        self.nn = nn

        self.x = WINDOW_WIDTH / 8
        self.y = WINDOW_HEIGHT / 2
        self.texture = pygame.transform.scale(pygame.image.load(BIRD_TEXTURE1).convert(), (BIRD_WIDTH, BIRD_HEIGHT))
        self.score = 0
        self.alive = True
        self._last = 1

    def update(self, jump):
        if self._last == 1:
            self.texture = pygame.transform.scale(pygame.image.load(BIRD_TEXTURE2).convert(),
                                                  (BIRD_WIDTH, BIRD_HEIGHT))
            self._last = 2
        elif self._last == 2:
            self.texture = pygame.transform.scale(pygame.image.load(BIRD_TEXTURE3).convert(),
                                                  (BIRD_WIDTH, BIRD_HEIGHT))
            self._last = 3
        elif self._last == 3:
            self.texture = pygame.transform.scale(pygame.image.load(BIRD_TEXTURE1).convert(),
                                                  (BIRD_WIDTH, BIRD_HEIGHT))
            self._last = 1

        # this whole ._last part only changes the texture, i know its the most inefficient way to do this

        if jump:
            self.y -= 100
        else:
            self.y += 20

    def draw(self):
        screen.blit(self.texture, (self.x, self.y))

    def get_data(self, pipe):
        return [self.x, self.y, pipe.x - self.x, pipe.y - self.y, speed]


pipes = []
birds = []

high_score = (0, 1)


def is_between(element, one, two):
    return (one >= element >= two) or (two >= element >= one)


def vec_is_between(element, one, two):
    return is_between(element[0], one[0], two[0]) and is_between(element[1], one[1], two[1])


current_gen = 0


def get_speed(score):
    return max(5, int(7 * math.log(max(score, 1), 5)))


def run(gen, config):
    global current_gen
    global high_score
    global speed

    for i, g in gen:
        nn = neat.nn.FeedForwardNetwork.create(g, config)
        g.fitness = 0
        birds.append(Bird(nn))

    current_gen += 1
    pipes.clear()
    speed = get_speed(0)
    while True:
        alive = 0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                sys.exit(0)
            if e.type == pygame.KEYUP: # to test without the nn
                if e.key == pygame.K_SPACE:
                    birds[0].update(True)
                    birds[0].draw()

        screen.blit(bg, (0, 0))
        for pipe in pipes:
            pipe.update()
            pipe.draw()
            if pipe.x <= -PIPE_WIDTH:
                pipes.clear()
                for bird in birds:
                    if bird.alive:
                        bird.score += 1

        if len(pipes) == 0:
            pipes.append(Pipe())

        biggest = 0

        pipeX1 = pipes[0].x
        pipeX2 = pipes[0].x + PIPE_WIDTH
        pipeY1 = pipes[0].y
        pipeY2 = pipes[0].y + PIPE_HEIGHT

        topPipeX1 = pipes[0].x
        topPipeX2 = pipes[0].x + PIPE_WIDTH
        topPipeY1 = 0
        topPipeY2 = pipes[0].y - PIPE_HEIGHT + WINDOW_HEIGHT / 4

        for bird in birds:
            if bird.alive:
                output = bird.nn.activate(bird.get_data(pipes[0]))
                choice = output.index(max(output))
                if choice == 0:
                    bird.update(False)
                elif choice == 1:
                    bird.update(True)

                bird.draw()
                if bird.y < 0:
                    bird.alive = False
                if bird.y > WINDOW_HEIGHT + BIRD_HEIGHT:
                    bird.alive = False

                birdX1 = bird.x
                birdX2 = bird.x + BIRD_WIDTH
                birdY1 = bird.y
                birdY2 = bird.y + BIRD_HEIGHT

                if ((is_between(birdX1, pipeX1, pipeX2) or is_between(birdX2, pipeX1, pipeX2)) and (
                        is_between(birdY1, pipeY1, pipeY2) or is_between(birdY2, pipeY1, pipeY2))) or (
                        is_between(birdX1, topPipeX1, topPipeX2) or is_between(birdX2, topPipeX1, topPipeX2)) and (
                        is_between(birdY1, topPipeY1, topPipeY2) or is_between(birdY2, topPipeY1, topPipeY2)):
                    bird.alive = False

                if bird.alive:
                    alive += 1
                if bird.score > biggest:
                    biggest = bird.score
                    speed = get_speed(biggest)
                    if biggest > high_score[0]:
                        high_score = (biggest, current_gen)

        screen.blit(font.render(f"Score: {biggest}", 1, (0, 0, 0)), (0, 0))
        screen.blit(font.render(f"Alive: {alive}", 1, (0, 0, 0)), (0, 24))
        screen.blit(font.render(f"Speed: {speed}", 1, (0, 0, 0)), (0, 48))
        screen.blit(font.render(f"Gen: {current_gen}", 1, (0, 0, 0)), (0, 72))
        screen.blit(font.render(f"HS: {high_score[0]} (Gen {high_score[1]})", 1, (0, 0, 0)), (0, 96))

        if alive == 0:
            for i, bird in enumerate(birds):
                try:
                    gen[i][1].fitness = bird.score
                except IndexError:
                    pass
            break

        pygame.display.flip()
        clock.tick(20)


if __name__ == "__main__":
    population = neat.Population(CONFIG)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-150')
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(1))
    winner = population.run(run, 1000)
