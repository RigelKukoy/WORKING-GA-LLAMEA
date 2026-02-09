from .funsearch import funsearch
from .llamea import LLaMEA
from .random_search import RandomSearch
from .lhns import LHNS_Method
from .mcts_ahd import MCTS_Method
from .ga_llamea import GA_LLaMEA_Method

try:
    from .eoh import EoH
    from .reevo import ReEvo
except Exception:  # pragma: no cover - optional dependency
    ReEvo = None
    EoH = None


__all__ = [
    "LLaMEA",
    "GA_LLaMEA_Method",
    "RandomSearch",
    "EoH",
    "funsearch",
    "ReEvo",
    "LHNS_Method",
    "MCTS_Method",
]
