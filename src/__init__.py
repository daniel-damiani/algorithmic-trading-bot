"""
QuantumSentiment Trading Bot

A sophisticated algorithmic trading system that combines multiple AI models
for market prediction and automated trading.
"""

__version__ = "1.0.0"
__author__ = "QuantumSentiment Team"

# Avoid eager imports of heavy ML subpackages at `import src` time.
__all__ = ['data', 'models', 'training']