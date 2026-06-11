from enum import Enum


class TemporalMode(str, Enum):
    NONE = "none"
    EARLY_FUSION = "early"
    RECURRENT = "recurrent"