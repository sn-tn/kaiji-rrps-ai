# Analysis Scripts

Run from the project root with `uv run -m analysis.<script>`.

---

## static.py

```bash
uv run -m analysis.static
uv run -m analysis.static --train 50000
uv run -m analysis.static --file my_agent
uv run -m analysis.static --load analysis/monty_hall_100000_0.999.pickle --gui
```

## tabular_nav.py

```bash
uv run -m analysis.tabular_nav
uv run -m analysis.tabular_nav --train 50000
uv run -m analysis.tabular_nav --file my_agent
uv run -m analysis.tabular_nav --load tabular_nav_20000_0.999.pickle --gui
```

## dqn.py

```bash
uv run -m analysis.dqn
uv run -m analysis.dqn --train 1000000
uv run -m analysis.dqn --file my_agent
uv run -m analysis.dqn --load analysis/dqn_nav --gui
```
