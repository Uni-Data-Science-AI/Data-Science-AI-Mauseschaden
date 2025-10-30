# Ranscht_detection
[miro](https://miro.com/welcomeonboard/SlFzT3RjNzNucjV3NFB1SjlxTmZNcVdCN0VUaG9pRWY0UEFUdXExUDFzdVpFeFJmSnpaN3NQWVNYMEI1NG0zTUwyYytlaCtVVStWcVlxMnkzUUNWbmM4U0s1UHZXTGdHUkU4UlAyWGV3ZWQwSmNIaUNJQkdpcytpZlVNQlBtcG5Bd044SHFHaVlWYWk0d3NxeHNmeG9BPT0hdjE=?share_link_id=35920731040)

[Report](https://urz365-my.sharepoint.com/:w:/g/personal/qp48reqo_uni-leipzig_de/EbHnH_fdEuZCmPb1fveyBxEBDaNVlK8tfu7Fl_nKpusG7g?e=CRbdQ3)


For setup refer to the [Installation guide](INSTALL.md).

# Project Structure

```bash
project/
├─ README.md
├─ INSTALL.md
├─ environment.yml
├─ notebooks/
│  ├─ 00_detection.ipynb
│  └─ 01_model_training.ipynb
├─ data/
│  └─ raw/
├─ models/        # ggf. in .gitignore aufnehmen, wenn groß
├─ runs/          # ggf. in .gitignore aufnehmen, wenn groß
└─ docs/
   └─ images/
```

### Commit Message Convention
- `[FEATURE]` - New feature
- `[FIX]` - Bug fix
- `[DOCS]` - Documentation changes
- `[STYLE]` - Code style changes (formatting, etc.)
- `[REFACTOR]` - Code refactoring
- `[TEST]` - Adding or modifying tests
- `[CHORE]` - Maintenance tasks


tmux

1. Neue tmux-Session starten
```bash
tmux new -s training
```

2. Dein Skript starten
```bash
python model_training.py
```

3. tmux-Session trennen (detach)
```bash
Ctrl + B, dann D
```
4. Später wieder verbinden
```bash
tmux attach -t training
```

Wenn du mehrere Sessions hast:
```bash
tmux ls
```