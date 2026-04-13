import pygame
import sys
import pygame_menu
import pandas as pd

WIDTH, HEIGHT = 1280, 720
FPS = 60

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RPS Arena")
clock = pygame.time.Clock()

# menu_width = 200
# menu_height = 100
# menu = pygame_menu.Menu(
#     "Cards",
#     menu_width,
#     menu_height,
#     position=(WIDTH - menu_width, 0, False),
# )


def render_table(
    screen,
    df,
    x=0,
    y=0,
    font=None,
    col_widths=None,
    row_height=28,
    header=True,
    bg_color=(20, 24, 34),
    header_bg_color=(30, 36, 52),
    border_color=(40, 48, 68),
    text_color=(220, 225, 235),
    header_text_color=(160, 170, 200),
):

    rows = [
        df.reset_index().columns.tolist()
    ] + df.reset_index().values.tolist()

    num_cols = len(rows[0]) if rows else 0
    if col_widths is None:
        col_w = screen.get_width() // num_cols
        col_widths = [col_w] * num_cols

    if font is None:
        font = pygame.font.SysFont("monospace", 14)
    for row_i, row in enumerate(rows):
        is_header = header and row_i == 0
        cx = x
        for col_i, cell in enumerate(row):
            col_w = col_widths[col_i]
            rect = pygame.Rect(cx, y + row_i * row_height, col_w, row_height)

            pygame.draw.rect(
                screen, header_bg_color if is_header else bg_color, rect
            )
            pygame.draw.rect(screen, border_color, rect, 1)

            color = header_text_color if is_header else text_color
            surf = font.render(str(cell), True, color)
            screen.blit(
                surf,
                (rect.x + 8, rect.y + (row_height - surf.get_height()) // 2),
            )

            cx += col_w


df = pd.DataFrame({"name": ["alice", "bob"], "age": [30, 25]})


while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()

    screen.fill((15, 15, 20))
    render_table(screen, df)
    pygame.display.flip()
    clock.tick(FPS)
