"""YAML-driven scenario loader used by the GUI and the CLI runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from ..envs import MyoArmConfig
from ..sensory import SensoryConfig


@dataclass
class Scenario:
    name: str
    controller: str = "pd_muscle"
    controller_kwargs: Dict[str, Any] = field(default_factory=dict)
    env: MyoArmConfig = field(default_factory=MyoArmConfig)
    sensory: SensoryConfig = field(default_factory=SensoryConfig)
    horizon_s: float = 6.0
    with_viewer: bool = True


def load_config(path: str | Path) -> Dict[str, Scenario]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    scenarios: Dict[str, Scenario] = {}
    for name, block in (raw.get("scenarios") or {}).items():
        env_kwargs = block.get("env", {}) or {}
        sens_kwargs = block.get("sensory", {}) or {}
        scenarios[name] = Scenario(
            name=name,
            controller=block.get("controller", "pd_muscle"),
            controller_kwargs=block.get("controller_kwargs", {}) or {},
            env=MyoArmConfig(**env_kwargs),
            sensory=SensoryConfig(**sens_kwargs),
            horizon_s=block.get("horizon_s", 6.0),
            with_viewer=bool(block.get("with_viewer", True)),
        )
    return scenarios
