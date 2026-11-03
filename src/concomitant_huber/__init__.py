"""Concomitant Huber regression with exact coefficient CD and scale MM."""
from .solver import HuberCD
from .fast_estimator import FastHuberCD
from .scale_updates import update_scale, ScaleUpdateResult
__version__ = '0.0.1'
__all__ = ['HuberCD', 'FastHuberCD', 'update_scale', 'ScaleUpdateResult']
