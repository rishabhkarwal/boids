import pygame, sys
import numpy as np
pygame.init()

clock = pygame.time.Clock()
fps = 60

WIDTH, HEIGHT = 500, 500
screen = pygame.display.set_mode([WIDTH, HEIGHT], pygame.NOFRAME)

background = (40, 44, 52)
accent = (255, 251, 237)

font = pygame.font.SysFont("consolas", 25)

class Agent:
    colour = (58, 124, 165) # main colour
    outline = (129, 195, 215)
    size = 10 # 'radius'
    
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def update(self):
        self.position += self.velocity

        if self.position[0] <= self.size or self.position[0] >= WIDTH - self.size: # if hits left/right
            self.velocity[0] *= -1 # reverse x-velocity

        if self.position[1] <= self.size or self.position[1] >= HEIGHT - self.size: # if hits top/bottom
            self.velocity[1] *= -1 # reverse y-velocity

    def draw(self):
        heading = np.arctan2(self.velocity[1], self.velocity[0])

        points = [
            self.position + self.size * np.array([np.cos(theta), np.sin(theta)])
            for theta in np.linspace(0, 2 * np.pi, 3 + 1)[:-1] + heading # gets angle offsets for n-gon
        ]
            
        pygame.draw.polygon(screen, self.colour, points)
        pygame.draw.polygon(screen, self.outline, points, 2)

def main():
    N = 3 # number of agents
 
    # arranges N agents in a circle moving outward
    boids = [
        Agent(np.array([WIDTH / 2, HEIGHT / 2]) + 10 * np.array([np.cos(theta), np.sin(theta)]), (np.cos(theta), np.sin(theta)))
        for theta in np.linspace(0, 2 * np.pi, N + 1)[:-1]
    ]

    while 1:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        screen.fill(background)

        for boid in boids:
            boid.update()
            boid.draw()

        screen.blit(font.render(f"FPS: {clock.get_fps() :.0f}", True, accent), (10, 10))

        pygame.display.update()

        clock.tick(fps)

if __name__ == '__main__':
    main()