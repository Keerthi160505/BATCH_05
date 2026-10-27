"""
Dynamic Green AI Evaluation Framework
A comprehensive framework to evaluate machine learning models
based on both prediction performance and environmental resource sustainability.
"""

from .data_pipeline import DynamicDataPipeline, MemoryBudgetExceededError
from .evaluator import GreenAIEvaluator
from .green_index import GreenAIIndexCalculator
from .models import ModelFactory
from .profiler import ResourceEnergyProfiler

__version__ = "1.0.0"
__author__ = "Green AI Research Initiative"

__all__ = [
    "DynamicDataPipeline",
    "MemoryBudgetExceededError",
    "GreenAIEvaluator",
    "GreenAIIndexCalculator",
    "ModelFactory",
    "ResourceEnergyProfiler"
]
