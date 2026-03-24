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

separation = 8
alignment = 4
cohesion = 2

# helper to normalise vector
def normalise(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude > 0:
        return vector / magnitude
    return vector

# helper to clamp vector to a set length
def clamp(vector, scale):
    magnitude = np.linalg.norm(vector)
    if magnitude > scale:
        return vector / magnitude * scale
    return vector

class Agent:
    colour = (58, 124, 165) # main colour
    outline = (129, 195, 215)
    size = 10 # 'radius'

    speed = 3.5 # max speed

    # radii of influence
    avoidance = 25
    vision = 50
    fov = np.radians(150)
    
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def update(self):
        self.velocity = clamp(self.velocity, self.speed)

        self.position += self.velocity

        # left wall
        if self.position[0] <= self.size:
            self.position[0] = self.size
            self.velocity[0] *= -1
        # right wall
        elif self.position[0] >= WIDTH - self.size:
            self.position[0] = WIDTH - self.size
            self.velocity[0] *= -1

        # top wall
        if self.position[1] <= self.size:
            self.position[1] = self.size
            self.velocity[1] *= -1
        # bottom wall
        elif self.position[1] >= HEIGHT - self.size:
            self.position[1] = HEIGHT - self.size
            self.velocity[1] *= -1

    def draw(self):
        heading = np.arctan2(self.velocity[1], self.velocity[0])

        points = [
            self.position + self.size * np.array([np.cos(theta), np.sin(theta)])
            for theta in np.radians([0, 140, 220]) + heading # sleeker design
        ]
            
        pygame.draw.polygon(screen, self.colour, points)
        pygame.draw.polygon(screen, self.outline, points, 2)

        # senses

        # radius of avoidance (separation)
        pygame.draw.circle(screen, (255, 20, 40), self.position, self.avoidance, 2)

        # radius of vision (alignment, cohesion)
        start_angle = -heading - self.fov / 2
        stop_angle = -heading + self.fov / 2

        pygame.draw.arc(screen, (25, 255, 68), (self.position[0] - self.vision, self.position[1] - self.vision, self.vision * 2, self.vision * 2), start_angle, stop_angle, 1)

        left_edge = self.position + self.vision * np.array([np.cos(-stop_angle), np.sin(-stop_angle)])
        right_edge = self.position + self.vision * np.array([np.cos(-start_angle), np.sin(-start_angle)])
        pygame.draw.line(screen, (25, 255, 68), self.position, left_edge, 1)
        pygame.draw.line(screen, (25, 255, 68), self.position, right_edge, 1)

    def sense(self, others):
        force = np.array([0.0, 0.0], dtype=float)
        _separation, _alignment, _cohesion = [], [], []
        
        heading = np.arctan2(self.velocity[1], self.velocity[0])

        for other in others:
            if self == other: continue

            direction = other.position - self.position
            distance = np.linalg.norm(direction)

            if distance > 0:
                angle = np.arctan2(direction[1], direction[0])
                angle -= heading
                
                # wrap the angle difference to remain between -pi and pi
                angle = (angle + np.pi) % (2 * np.pi) - np.pi

                in_view = abs(angle) < (self.fov / 2)

                # separation applies all the way around
                if distance < self.avoidance:
                    _separation.append(-direction / distance)
                
                # alignment and cohesion only apply if the neighbour is visible
                if in_view and distance < self.vision: 
                    _alignment.append(other.velocity)
                    _cohesion.append(other.position)

        # separation
        if _separation:
            force += normalise(np.mean(_separation, axis=0)) * separation
            
        # alignment
        if _alignment:
            # steer towards the average velocity
            steer = np.mean(_alignment, axis=0) - self.velocity
            force += normalise(steer) * alignment

        # cohesion
        if _cohesion:
            # vector pointing from agent to the centre of mass
            steer = np.mean(_cohesion, axis=0) - self.position
            force += normalise(steer) * cohesion

        # so they do not snap instantly
        max_force = 0.15
        force = clamp(force, max_force)

        self.velocity += force

def main():
    N = 50 # number of agents
 
    # arranges N agents in a circle moving outward
    boids = [
        Agent(np.array([WIDTH / 2, HEIGHT / 2]) + 100 * np.array([np.cos(theta), np.sin(theta)]), (np.cos(theta), np.sin(theta)))
        for theta in np.linspace(0, 2 * np.pi, N + 1)[:-1]
    ]

    while 1:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        screen.fill(background)

        for boid in boids:
            boid.sense(boids)
            boid.update()
            boid.draw()

        screen.blit(font.render(f"FPS: {clock.get_fps() :.0f}", True, accent), (10, 10))

        pygame.display.update()

        clock.tick(fps)

if __name__ == '__main__':
    main()