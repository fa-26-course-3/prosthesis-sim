"""Controller protocol shared by every policy in the stack."""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ..sensory import Observation


class BaseController(Protocol):
    """All controllers accept a filtered :class:`Observation` and produce muscle activations."""

    name: str

    def reset(self) -> None: ...

    def act(self, obs: Observation, target: Optional[np.ndarray] = None) -> np.ndarray:
        """Return an activation vector shape (6,) in [0, 1]."""
        ...
