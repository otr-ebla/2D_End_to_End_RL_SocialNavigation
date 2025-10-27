"""2D social navigation simulation environment."""

from __future__ import annotations

from typing import Any, Callable

from .environment import EnvironmentConfig, SocialNavigationEnv

keyboard_control: Callable[..., Any]

try:  # pragma: no cover - defensive safeguard.
    from .environment import keyboard_control as _keyboard_control
except (ImportError, AttributeError) as exc:
    def _missing_keyboard_control(*_: Any, **__: Any) -> None:
        raise ImportError(
            "keyboard_control is not available. Ensure you pulled the latest code "
            "and that optional dependencies such as curses are installed."
        ) from exc

    keyboard_control = _missing_keyboard_control
else:
    keyboard_control = _keyboard_control

__all__ = ["SocialNavigationEnv", "EnvironmentConfig", "keyboard_control"]
from . import environment as _environment

EnvironmentConfig = _environment.EnvironmentConfig
SocialNavigationEnv = _environment.SocialNavigationEnv

keyboard_control: Callable[..., Any]

try:
    keyboard_control = _environment.keyboard_control
except AttributeError as exc:  # pragma: no cover - defensive safeguard.
    def _missing_keyboard_control(*_: Any, **__: Any) -> None:
        raise ImportError(
            "keyboard_control is not available. Ensure you are running a version of "
            "social_nav_env that includes the teleoperation helper."
        ) from exc

    keyboard_control = _missing_keyboard_control

__all__ = ["SocialNavigationEnv", "EnvironmentConfig", "keyboard_control"]
from .environment import SocialNavigationEnv, EnvironmentConfig

__all__ = ["SocialNavigationEnv", "EnvironmentConfig"]
