"""
QuantumSentiment Trading Bot

A sophisticated algorithmic trading system that combines multiple AI models
for market prediction and automated trading.
"""

__version__ = "1.0.0"
__author__ = "QuantumSentiment Team"

# Core modules
from . import data
from . import models
from . import training

__all__ = ['data', 'models', 'training']