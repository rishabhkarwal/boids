import pygame, sys
pygame.init()

clock = pygame.time.Clock()
fps = 60

WIDTH, HEIGHT = 500, 500
screen = pygame.display.set_mode([WIDTH, HEIGHT], pygame.NOFRAME)

background = (40, 44, 52)
accent = (255, 251, 237)

font = pygame.font.SysFont("consolas", 25)

def main():
    while 1:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

        screen.fill(background)

        screen.blit(font.render(f"FPS: {clock.get_fps() :.0f}", True, accent), (10, 10))

        pygame.display.update()

        clock.tick(fps)

if __name__ == '__main__':
    main()