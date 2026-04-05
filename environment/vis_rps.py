import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pygame
import time
from environment.rps_gym import RestrictedRPSEnv
from environment.move import Move

CELL_SIZE = 40
CONSOLE_WIDTH = 350

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
BLUE = (0, 0, 255)
DARK_TURQUOISE = (0, 150, 150)
BURGUNDY = (128, 0, 32)
HIGHLIGHT = (255, 255, 180)  # challenge-radius cells

screen = None
game_ended = False
action_results = []
clock = None

game = RestrictedRPSEnv()


# ── helpers ───────────────────────────────────────────────────────────────────


def _grid_px():
    return CELL_SIZE * game.grid_size


def _pos_to_pixel(pos: tuple[int, int]) -> tuple[int, int]:
    """Grid (x, y) → top-left pixel of that cell."""
    return pos[0] * CELL_SIZE, pos[1] * CELL_SIZE


# ── setup ─────────────────────────────────────────────────────────────────────


def setup(GUI: bool = True, env: RestrictedRPSEnv | None = None):
    global screen, game
    if env is not None:
        game = env
    if GUI:
        pygame.init()
        width = _grid_px() + CONSOLE_WIDTH
        height = _grid_px()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Restricted RPS Visualizer")


# ── drawing ───────────────────────────────────────────────────────────────────


def draw_grid():
    gp = _grid_px()
    for x in range(0, gp, CELL_SIZE):
        for y in range(0, gp, CELL_SIZE):
            pygame.draw.rect(
                screen, BLACK, pygame.Rect(x, y, CELL_SIZE, CELL_SIZE), 1
            )
    pygame.draw.rect(screen, GRAY, pygame.Rect(gp, 0, CONSOLE_WIDTH, gp))


def draw_challenge_radius(agent_pos: tuple[int, int]):
    """Highlight cells within the agent's challenge radius."""
    r = game.challenge_radius
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            cx, cy = agent_pos[0] + dx, agent_pos[1] + dy
            if 0 <= cx < game.grid_size and 0 <= cy < game.grid_size:
                px, py = _pos_to_pixel((cx, cy))
                pygame.draw.rect(
                    screen,
                    HIGHLIGHT,
                    pygame.Rect(px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2),
                )


def draw_agent(pos: tuple[int, int]):
    px, py = _pos_to_pixel(pos)
    cx, cy = px + CELL_SIZE // 2, py + CELL_SIZE // 2
    pygame.draw.circle(screen, DARK_TURQUOISE, (cx, cy), CELL_SIZE // 3)
    label = pygame.font.Font(None, 20).render("A", True, WHITE)
    screen.blit(label, label.get_rect(center=(cx, cy)))


def draw_opponents():
    for op in game._opponents:
        if not op.is_alive():
            continue
        px, py = _pos_to_pixel(op.position)
        cx, cy = px + CELL_SIZE // 2, py + CELL_SIZE // 2
        pygame.draw.circle(screen, BURGUNDY, (cx, cy), CELL_SIZE // 3)
        label = pygame.font.Font(None, 20).render(str(op.id), True, WHITE)
        screen.blit(label, label.get_rect(center=(cx, cy)))


def _wrap_text(font, text: str, max_width: int) -> list[str]:
    lines = []
    line = ""
    for word in text.split():
        test = line + word + " "
        if font.size(test)[0] > max_width:
            lines.append(line.strip())
            line = word + " "
        else:
            line = test
    if line.strip():
        lines.append(line.strip())
    return lines


def draw_console(reward: float, info: dict):
    ag = game._agent
    gp = _grid_px()
    cx = gp + 10
    y = 10
    max_w = CONSOLE_WIDTH - 25

    f_title = pygame.font.Font(None, 30)
    f_info = pygame.font.Font(None, 22)
    f_small = pygame.font.Font(None, 18)

    # ── stats ─────────────────────────────────────────────────────────────────
    screen.blit(f_title.render("Game State", True, BLUE), (cx, y))
    y += 35

    if ag:
        alive_ops = sum(1 for p in game._opponents if p.is_alive())
        in_range = sum(
            1
            for p in game._opponents
            if p.is_alive() and game._in_range(ag, p)
        )
        info_lines = [
            f"• Lives:    {ag.lives}",
            f"• Position: {ag.position}",
            (
                f"• Budget    R:{ag.budget[Move.ROCK]} "
                f" P:{ag.budget[Move.PAPER]}  S:{ag.budget[Move.SCISSORS]}"
            ),
            f"• Opponents alive: {alive_ops}",
            f"• In range:        {in_range}",
            f"• Last reward: {reward:+.1f}",
        ]
        if info.get("result"):
            info_lines.append(f"• Result: {info['result']}")
        for line in info_lines:
            screen.blit(f_info.render(line, True, BLACK), (cx, y))
            y += 26

    y += 8
    pygame.draw.line(screen, BLACK, (cx, y), (cx + CONSOLE_WIDTH - 20, y), 2)
    y += 12

    # ── legend ────────────────────────────────────────────────────────────────
    f_leg = pygame.font.Font(None, 20)
    pygame.draw.circle(screen, DARK_TURQUOISE, (cx + 8, y + 8), 8)
    screen.blit(f_leg.render("Agent (A)", True, BLACK), (cx + 22, y))
    y += 22
    pygame.draw.circle(screen, BURGUNDY, (cx + 8, y + 8), 8)
    screen.blit(f_leg.render("Opponent (ID)", True, BLACK), (cx + 22, y))
    y += 22
    pygame.draw.rect(screen, HIGHLIGHT, pygame.Rect(cx, y, 16, 16))
    pygame.draw.rect(screen, BLACK, pygame.Rect(cx, y, 16, 16), 1)
    screen.blit(f_leg.render("Challenge range", True, BLACK), (cx + 22, y))
    y += 26

    pygame.draw.line(screen, BLACK, (cx, y), (cx + CONSOLE_WIDTH - 20, y), 2)
    y += 12

    # ── controls ──────────────────────────────────────────────────────────────
    screen.blit(f_info.render("Controls:", True, BLUE), (cx, y))
    y += 24
    for c in [
        "WASD: Move",
        "R: Rock  P: Paper  K: Scissors",
        "Space: Restart    Esc: Quit",
    ]:
        screen.blit(f_leg.render(c, True, DARK_GRAY), (cx, y))
        y += 18
    y += 8

    pygame.draw.line(screen, BLACK, (cx, y), (cx + CONSOLE_WIDTH - 20, y), 2)
    y += 12

    # ── recent actions ────────────────────────────────────────────────────────
    screen.blit(
        pygame.font.Font(None, 26).render("Recent Actions:", True, BLUE),
        (cx, y),
    )
    y += 28

    for entry in action_results[-8:]:
        for wrapped_line in _wrap_text(f_small, entry, max_w):
            screen.blit(f_small.render(wrapped_line, True, BLACK), (cx, y))
            y += 18
        y += 4


def display_end_message(message: str):
    font = pygame.font.Font(None, 80)
    surf = font.render(message, True, DARK_GRAY)
    rect = surf.get_rect(center=(_grid_px() // 2, _grid_px() // 2))
    bg = pygame.Surface((rect.width + 20, rect.height + 10))
    bg.fill(WHITE)
    bg.set_alpha(200)
    screen.blit(bg, (rect.x - 10, rect.y - 5))
    screen.blit(surf, rect)


def _render_frame(reward: float, info: dict):
    screen.fill(WHITE)
    draw_grid()
    if game._agent:
        draw_challenge_radius(game._agent.position)
        draw_opponents()
        draw_agent(game._agent.position)
    draw_console(reward, info)


# ── interactive main loop ─────────────────────────────────────────────────────


def main():
    global game_ended, action_results, clock
    setup()
    clock = pygame.time.Clock()
    _, _ = game.reset()
    reward = 0.0
    info: dict = {}
    end_message = ""
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    _, _ = game.reset()
                    reward, info = 0.0, {}
                    action_results.clear()
                    game_ended = False
                    end_message = ""

                if game_ended:
                    continue

                key_to_action = {
                    pygame.K_w: (0, "Move Up"),
                    pygame.K_s: (1, "Move Down"),
                    pygame.K_a: (2, "Move Left"),
                    pygame.K_d: (3, "Move Right"),
                    pygame.K_r: (4, "ROCK"),
                    pygame.K_p: (5, "PAPER"),
                    pygame.K_k: (6, "SCISSORS"),
                }
                if event.key in key_to_action:
                    action, label = key_to_action[event.key]
                    _, reward, terminated, _, info = game.step(action)
                    entry = label
                    if reward != 0:
                        entry += f"  R:{reward:+.0f}"
                    if info.get("result"):
                        entry += f"  [{info['result']}]"
                    action_results.append(entry)
                    if terminated:
                        game_ended = True
                        end_message = (
                            "Victory!"
                            if info.get("result") == "victory"
                            else "Eliminated!"
                        )

        _render_frame(reward, info)
        if game_ended:
            display_end_message(end_message)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


# ── agent/training refresh ────────────────────────────────────────────────────


def refresh(_obs, reward: float, _done: bool, info: dict, delay: float = 0.1):
    """Call from a training loop to animate each step."""
    global clock, game_ended, action_results

    label = info.get("action", "step")
    entry = str(label)
    if reward != 0:
        entry += f"  R:{reward:+.0f}"
    if info.get("result"):
        entry += f"  [{info['result']}]"
    action_results.append(entry)
    if len(action_results) > 10:
        action_results.pop(0)

    if clock is None:
        clock = pygame.time.Clock()

    _render_frame(reward, info)

    if info.get("result") in ("victory", "eliminated"):
        display_end_message(
            "Victory!" if info["result"] == "victory" else "Eliminated!"
        )

    pygame.display.flip()
    clock.tick(60)
    time.sleep(delay)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

           
if __name__ == "__main__":
    main()
