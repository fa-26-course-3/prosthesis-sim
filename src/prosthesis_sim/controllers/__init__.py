from .base import BaseController
from .scripted import ScriptedController
from .pd_muscle import PDMuscleController
from .motornet_policy import MotorNetPolicyController
from .webcam_teleop import WebcamTeleopController

__all__ = [
    "BaseController",
    "ScriptedController",
    "PDMuscleController",
    "MotorNetPolicyController",
    "WebcamTeleopController",
    "make",
]


def make(name: str, env=None, **kwargs) -> BaseController:
    """Factory used by the config loader / GUI.

    All controllers can be built via ``from_env(env, **kwargs)`` when the env
    is available, or by explicit kwargs otherwise.
    """
    name = name.lower()
    if env is not None:
        if name in ("scripted", "open_loop", "arm_heben"):
            return ScriptedController.from_env(env, **kwargs)
        if name in ("pd", "pd_muscle"):
            return PDMuscleController.from_env(env, **kwargs)
        if name in ("motornet", "policy"):
            return MotorNetPolicyController.from_env(env, **kwargs)
        if name in ("webcam", "teleop", "webcam_teleop"):
            return WebcamTeleopController.from_env(env, **kwargs)
    else:
        if name in ("scripted", "open_loop", "arm_heben"):
            return ScriptedController(**kwargs)
        if name in ("pd", "pd_muscle"):
            return PDMuscleController(**kwargs)
        if name in ("motornet", "policy"):
            return MotorNetPolicyController(**kwargs)
        if name in ("webcam", "teleop", "webcam_teleop"):
            return WebcamTeleopController(**kwargs)
    raise ValueError(f"unknown controller '{name}'")
