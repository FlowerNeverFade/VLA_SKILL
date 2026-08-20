from __future__ import annotations

from pathlib import Path

import pytest

from .helpers import make_toy_skill


@pytest.fixture()
def toy_skill_root(tmp_path: Path) -> Path:
    make_toy_skill(
        tmp_path,
        skill_id="pick_mug",
        display_name="Pick Mug",
        task="pick the mug and place it on the tray",
        aliases=["pick mug"],
        keywords=["mug", "tray"],
        regexes=["pick .* mug"],
        priority=10,
        seed=1,
    )
    make_toy_skill(
        tmp_path,
        skill_id="open_drawer",
        display_name="Open Drawer",
        task="open the drawer fully",
        aliases=["open drawer"],
        keywords=["drawer"],
        regexes=["open .* drawer"],
        priority=5,
        seed=2,
    )
    return tmp_path
