import pygame
from pathlib import Path

from rrps_core.cards import Card

_surface = None
_clock = None
_match_log = [] 
_star_img = None
_star_loaded = False
MAX_LOG = 10

# game win/loss counters
_game_wins = 0
_game_losses = 0
_game_total = 0

_STAR_PATH = Path(__file__).parent / "assets" / "star.png"
STAR_SIZE = 22 

CELL_SIZE = 60
PADDING = 40

AGENT_COLOR = (50, 220, 80)
OPP_COLOR = (180, 100, 100)
BG_COLOR = (15, 15, 20)
GRID_COLOR = (40, 48, 68)
TEXT_COLOR = (220, 225, 235)
DIM_COLOR = (100, 110, 135)
WIN_COLOR = (80, 210, 100)
LOSE_COLOR = (210, 80, 80)
TIE_COLOR = (200, 180, 60)

_WINS_AGAINST = {
    Card.rock: Card.scissors,
    Card.paper: Card.rock,
    Card.scissors: Card.paper,
}


def _load_star():
    global _star_img, _star_loaded
    if _star_loaded:
        return
    _star_loaded = True
    try:
        img = pygame.image.load(str(_STAR_PATH)).convert_alpha()
        _star_img = pygame.transform.smoothscale(img, (STAR_SIZE, STAR_SIZE))
    except Exception as e:
        print(f"[grid_view] could not load star image: {e}")


def init(grid_rows, grid_cols):
    global _surface, _clock
    if not pygame.get_init():
        pygame.init()
    width = grid_cols * CELL_SIZE + PADDING * 2
    height = grid_rows * CELL_SIZE + PADDING * 2
    _surface = pygame.display.set_mode((width, height), pygame.SHOWN)
    pygame.display.set_caption("RPS Arena — Grid View")
    _clock = pygame.time.Clock()


def update_match_log(terminated, info):
    """Call once per step (from visualizer.refresh) to record agent matchups and game outcomes."""
    global _match_log, _game_wins, _game_losses, _game_total
    matchup_dict = info.get("matchup_dict") or {}
    turn = info.get("round_number", 0)

    for (pid1, pid2), (card1, card2) in matchup_dict.items():
        if pid1 != 0 and pid2 != 0:
            continue
        agent_card, opp_card, opp_id = (
            (card1, card2, pid2) if pid1 == 0 else (card2, card1, pid1)
        )
        if agent_card == opp_card:
            result = "Tie"
        elif _WINS_AGAINST[agent_card] == opp_card:
            result = "Won"
        else:
            result = "Lost"
        _match_log.append((turn, opp_id, agent_card, opp_card, result))

    if len(_match_log) > MAX_LOG:
        _match_log = _match_log[-MAX_LOG:]

    if terminated:
        status = info.get("game_status")
        _game_total += 1
        if status == "victory":
            _game_wins += 1
        elif status == "eliminated":
            _game_losses += 1


def draw_to(surface, x_off, y_off, alive_player_dict, grid_rows, grid_cols):
    """Draw the grid and info panel onto *surface* at pixel offset (x_off, y_off)."""
    _load_star()
    font = pygame.font.SysFont("monospace", 16)
    small = pygame.font.SysFont("monospace", 13)
    stars_font = pygame.font.SysFont("monospace", 26)

    # map 
    pos_map = {}
    for pid, p in alive_player_dict.items():
        key = (int(p["position"][0]), int(p["position"][1]))
        if key not in pos_map or pid == 0:
            pos_map[key] = (pid, p)

    # draw grid
    for r in range(grid_rows):
        for c in range(grid_cols):
            x = x_off + PADDING + c * CELL_SIZE
            y = y_off + PADDING + r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            if (r, c) in pos_map:
                pid, _ = pos_map[(r, c)]
                color = AGENT_COLOR if pid == 0 else OPP_COLOR
                pygame.draw.rect(surface, color, rect, border_radius=8)
                pygame.draw.rect(surface, GRID_COLOR, rect, 1, border_radius=8)
                label = font.render(str(pid), True, (255, 255, 255))
                surface.blit(label, (rect.centerx - label.get_width() // 2,
                                     rect.centery - label.get_height() // 2))
            else:
                pygame.draw.rect(surface, BG_COLOR, rect)
                pygame.draw.rect(surface, GRID_COLOR, rect, 1)

    # panel below grid
    panel_x = x_off + PADDING
    panel_y = y_off + PADDING + grid_rows * CELL_SIZE + 12

    # agent stars + win/loss rate
    agent = alive_player_dict.get(0)
    if agent is not None:
        wr = f"{_game_wins/_game_total*100:.0f}%"   if _game_total else "—"
        lr = f"{_game_losses/_game_total*100:.0f}%" if _game_total else "—"

        label_surf = stars_font.render("Agent", True, AGENT_COLOR)
        count_surf = stars_font.render(str(agent["stars_total"]), True, TEXT_COLOR)
        wr_surf    = stars_font.render(f"W:{wr}", True, WIN_COLOR)
        lr_surf    = stars_font.render(f"L:{lr}", True, LOSE_COLOR)

        row_h = max(s.get_height() for s in [label_surf, count_surf, wr_surf, lr_surf])
        cy = panel_y + row_h // 2
        x = panel_x

        surface.blit(label_surf, (x, cy - label_surf.get_height() // 2))
        x += label_surf.get_width() + 8

        if _star_img is not None:
            surface.blit(_star_img, (x, cy - _star_img.get_height() // 2))
            x += _star_img.get_width() + 4

        surface.blit(count_surf, (x, cy - count_surf.get_height() // 2))
        x += count_surf.get_width() + 16

        surface.blit(wr_surf, (x, cy - wr_surf.get_height() // 2))
        x += wr_surf.get_width() + 10

        surface.blit(lr_surf, (x, cy - lr_surf.get_height() // 2))

        panel_y += row_h + 10

    # agent match log
    if _match_log:
        hdr = small.render("Recent matches:", True, DIM_COLOR)
        surface.blit(hdr, (panel_x, panel_y))
        panel_y += 20

        for turn, opp_id, agent_card, opp_card, result in reversed(_match_log):
            color = WIN_COLOR if result == "Won" else (LOSE_COLOR if result == "Lost" else TIE_COLOR)
            line = (f"T{turn:>3}  vs P{opp_id}  "
                    f"{agent_card.name[0].upper()} vs {opp_card.name[0].upper()}  "
                    f"{result}")
            surf = small.render(line, True, color)
            surface.blit(surf, (panel_x, panel_y))
            panel_y += 18


def render(alive_player_dict, grid_rows, grid_cols):
    """only used when grid_view owns its own window."""
    if _surface is None:
        return
    pygame.event.pump()
    _surface.fill(BG_COLOR)
    draw_to(_surface, 0, 0, alive_player_dict, grid_rows, grid_cols)
    pygame.display.flip()
    _clock.tick(60)
