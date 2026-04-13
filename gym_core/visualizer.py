import pygame
import sys
import pygame_menu
import pandas as pd
from gym_core.info import Info

FPS = 60

screen = None
clock = None
control_menu = None
next_round_menu = None

cell_rects = {}
cell_menu = None
autoplay = False

next_round = False


def init(width=1280, height=1000):
    global screen, clock, control_menu, next_round_menu

    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.SHOWN)
    pygame.display.set_caption("RPS Arena")
    clock = pygame.time.Clock()

    control_theme = pygame_menu.Theme(
        background_color=(20, 24, 34),
        title=False,
        widget_font=pygame_menu.font.FONT_8BIT,
        widget_font_size=12,
        widget_font_color=(220, 225, 235),
        widget_background_color=(30, 36, 52),
        widget_border_width=1,
        widget_border_color=(40, 48, 68),
    )

    control_menu = pygame_menu.Menu(
        "",
        200,
        60,
        position=(width - 200, 0, False),
        theme=control_theme,
    )
    control_menu.add.button(
        f"autoplay: {'ON' if autoplay else 'OFF'}", _toggle_autoplay_btn
    )

    next_round_menu = pygame_menu.Menu(
        "",
        200,
        60,
        position=(width - 200, 60, False),
        theme=control_theme,
    )
    next_round_menu.add.button("Next Round", _on_next_round)


def _on_next_round():
    global next_round
    next_round = True


def _toggle_autoplay_btn():
    global autoplay
    autoplay = not autoplay
    btn = control_menu.get_widgets()[0]
    btn.set_title(f"autoplay: {'ON' if autoplay else 'OFF'}")


def make_cell_menu(screen, row_i, col_i, rect):
    """Create a popup menu anchored to the clicked cell."""
    menu = pygame_menu.Menu(
        f"Row {row_i} Col {col_i}",
        200,
        150,
        position=(rect.x, rect.bottom, False),
    )
    menu.add.button("Option A", lambda: print("option a"))
    menu.add.button("Option B", lambda: print("option b"))
    menu.add.button("Close", lambda: None)
    return menu


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
    highlight_rows=None,
):
    highlight_rows = (
        set(df.index[df["player_id"] == 0].tolist())
        if "player_id" in df
        else set()
    )
    cell_rects = {}

    rows = [df.columns.tolist()] + df.values.tolist()

    num_cols = len(rows[0]) if rows else 0
    if col_widths is None:
        col_w = screen.get_width() // num_cols
        col_widths = [col_w] * num_cols

    if font is None:
        font = pygame.font.SysFont("monospace", 14)
    for row_i, row in enumerate(rows):
        is_header = header and row_i == 0
        is_highlight = not is_header and (row_i - 1) in highlight_rows

        cx = x
        fill = bg_color
        for col_i, cell in enumerate(row):
            col_w = col_widths[col_i]
            rect = pygame.Rect(cx, y + row_i * row_height, col_w, row_height)
            cell_rects[(row_i, col_i)] = rect
            if is_header:
                fill = header_bg_color
            elif is_highlight:
                fill = (0, 255, 80)
            else:
                fill = bg_color
            pygame.draw.rect(screen, fill, rect)
            pygame.draw.rect(screen, border_color, rect, 1)

            color = text_color
            if is_header:
                color = header_text_color
            elif is_highlight:
                color = (0, 0, 0)

            surf = font.render(str(cell), True, color)
            screen.blit(
                surf,
                (rect.x + 8, rect.y + (row_height - surf.get_height()) // 2),
            )

            cx += col_w
    return cell_rects


def handle_events(events):
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()


def wait_for_click():
    global autoplay, next_round
    next_round = False

    def _draw_menus(events):
        handle_events(events)
        control_menu.update(events)
        control_menu.draw(screen)
        next_round_menu.update(events)
        next_round_menu.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    if autoplay:
        start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start < 500:
            events = pygame.event.get()
            _draw_menus(events)
            if not autoplay:  # toggled off mid-wait, fall through to manual
                break
        if autoplay:
            return

    while not next_round:
        events = pygame.event.get()
        _draw_menus(events)


def toggle_autoplay():
    global autoplay
    autoplay = not autoplay


def refresh(terminated: bool, truncated: bool, info: Info):
    screen.fill((15, 15, 20))
    y = 10
    font = pygame.font.SysFont("monospace", 14)
    # ── round number ──────────────────────────────────────────────────
    surf = font.render(f"Round {info['round_number']}", True, (220, 225, 235))
    screen.blit(surf, (10, y))
    y += 28

    # ── initial_alive_player_dict ─────────────────────────────────────────────
    surf = font.render(f"Start Alive Player Dict", True, (220, 225, 235))
    screen.blit(surf, (10, y))
    y += 28
    initial_alive_player_dict = info["initial_alive_player_dict"]
    if initial_alive_player_dict:
        player_df = pd.DataFrame(initial_alive_player_dict).T
        player_df.index.name = "player_id"
        player_df = player_df.reset_index()
        render_table(screen, player_df, y=y)
        y += (len(player_df) + 1) * 28 + 10
    else:
        screen.blit(
            font.render("players: skipped", True, (100, 110, 135)), (10, y)
        )
        y += 28

    # ── challenge_table ───────────────────────────────────────────────
    surf = font.render(f"Challenge Table", True, (220, 225, 235))
    screen.blit(surf, (10, y))
    y += 28
    challenge_table = info["challenge_table"]
    if challenge_table is not None and not challenge_table.empty:
        render_table(screen, challenge_table, y=y)
        y += (len(challenge_table) + 1) * 28 + 10
    else:
        screen.blit(
            font.render("challenges: skipped", True, (100, 110, 135)), (10, y)
        )
        y += 28

    # ── matchup_dict ──────────────────────────────────────────────────
    surf = font.render(f"Matchup Table", True, (220, 225, 235))
    screen.blit(surf, (10, y))
    y += 28
    matchup_dict = info["matchup_dict"]
    if matchup_dict:
        matchup_df = pd.DataFrame(
            [(p1, p2, c1, c2) for (p1, p2), (c1, c2) in matchup_dict.items()],
            columns=["player_1", "player_2", "card_1", "card_2"],
        )
        render_table(screen, matchup_df, y=y)
        y += (len(matchup_df) + 1) * 28 + 10
    else:
        screen.blit(
            font.render("matchups: skipped", True, (100, 110, 135)), (10, y)
        )
        y += 28
    # ── alive_player_dict ─────────────────────────────────────────────
    surf = font.render(f"End Alive Player Dict", True, (220, 225, 235))
    screen.blit(surf, (10, y))
    y += 28
    alive_player_dict = info["alive_player_dict"]
    if alive_player_dict:
        alive_player_dict = pd.DataFrame(alive_player_dict).T
        alive_player_dict.index.name = "player_id"
        alive_player_dict = player_df.reset_index()
        render_table(screen, player_df, y=y)
    else:
        screen.blit(
            font.render("players: skipped", True, (100, 110, 135)), (10, y)
        )

    control_menu.draw(screen)
    next_round_menu.draw(screen)
    pygame.display.flip()

    wait_for_click()
