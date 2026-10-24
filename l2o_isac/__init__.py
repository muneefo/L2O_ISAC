"""Independent, equation-based implementation of Wang, Vorobyov & Ollila (2026)."""
from .config import SystemConfig
from .data import ChannelBatch, generate_channels, initialize_state
from .physics import metrics, gradients
from .solver import PGDA, UnrolledPGDA, SolverState
from .networks import EpsNet

__all__ = ["SystemConfig", "ChannelBatch", "generate_channels", "initialize_state",
           "metrics", "gradients", "PGDA", "UnrolledPGDA", "SolverState", "EpsNet"]
