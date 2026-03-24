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

class Grid:
    def __init__(self, size):
        self.size = size
        self.squares = {}
        # precompute the 9 spatial offsets
        self.offsets = [
            (dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
        ]

    def clear(self):
        self.squares.clear()

    def convert(self, position):
        return (int(position[0] // self.size), int(position[1] // self.size))

    # drop an agent into its current square
    def insert(self, agent):
        square = self.convert(agent.position)
        if square not in self.squares:
            self.squares[square] = []
        self.squares[square].append(agent)

    # retrieve all agents from a specific square and its 8 neighbours
    def get_neighbours(self, position):
        x, y = self.convert(position)
        neighbours = []
        
        # loop through a 3x3 grid around the boid
        for dx, dy in self.offsets:
            square = (x + dx, y + dy)
            if square in self.squares:
                neighbours.extend(self.squares[square])
                    
        return neighbours

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
        # right wall
        elif self.position[0] >= WIDTH - self.size:
            self.position[0] = WIDTH - self.size

        # top wall
        if self.position[1] <= self.size:
            self.position[1] = self.size
        # bottom wall
        elif self.position[1] >= HEIGHT - self.size:
            self.position[1] = HEIGHT - self.size

    def draw(self):
        heading = np.arctan2(self.velocity[1], self.velocity[0])

        points = [
            self.position + self.size * np.array([np.cos(theta), np.sin(theta)])
            for theta in np.radians([0, 140, 220]) + heading # sleeker design
        ]
            
        pygame.draw.polygon(screen, self.colour, points)
        pygame.draw.polygon(screen, self.outline, points, 2)

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

        # wall avoidance
        _wall = np.array([0.0, 0.0], dtype=float)
        distance = self.avoidance * 2
        
        # left wall
        dist_left = self.position[0]
        if dist_left < distance:
            _wall[0] += 1 / max(dist_left, 0.1)
            
        # right wall
        right = WIDTH - self.position[0]
        if right < distance:
            _wall[0] -= 1 / max(right, 0.1)
            
        # top wall
        top = self.position[1]
        if top < distance:
            _wall[1] += 1 / max(top, 0.1)
            
        # bottom wall
        bottom = HEIGHT - self.position[1]
        if bottom < distance:
            _wall[1] -= 1 / max(bottom, 0.1)
            
        if np.linalg.norm(_wall) > 0:
            force += normalise(_wall) * 1000 # very high to prevent piercing

        # so they do not snap instantly
        max_force = 0.15
        force = clamp(force, max_force)

        self.velocity += force

def main():
    N = 100 # number of agents
 
    # arranges N agents in a circle moving outward
    boids = [
        Agent(np.array([WIDTH / 2, HEIGHT / 2]) + 100 * np.array([np.cos(theta), np.sin(theta)]), (np.cos(theta), np.sin(theta)))
        for theta in np.linspace(0, 2 * np.pi, N + 1)[:-1]
    ]

    grid = Grid(max(Agent.vision, Agent.avoidance))

    while 1:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        screen.fill(background)

        grid.clear()
        for boid in boids:
            grid.insert(boid)

        for boid in boids:
            boid.sense(grid.get_neighbours(boid.position))
            boid.update()
            boid.draw()

        screen.blit(font.render(f"FPS: {clock.get_fps() :.0f}", True, accent), (10, 10))

        pygame.display.update()

        clock.tick(fps)

if __name__ == '__main__':
    main()