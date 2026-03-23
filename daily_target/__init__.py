"""
Daily Target — Compound profit management system.

Coordinates all modules toward achieving a configurable daily target
while protecting the compound base. Manages trading modes:
NORMAL, AGGRESSIVE, PROTECTING, HALTED.
"""

from daily_target.tracker import DailyTargetTracker, DailyTargetState, DailyTargetContext, TradingMode
from daily_target.mode_controller import ModeController
from daily_target.compounder import Compounder

__all__ = [
    "DailyTargetTracker",
    "DailyTargetState",
    "DailyTargetContext",
    "TradingMode",
    "ModeController",
    "Compounder",
]
