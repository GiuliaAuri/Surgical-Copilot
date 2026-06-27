from enum import Enum


class TemporalMode(str, Enum):
    NONE = "none"
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    RECURRENT = "recurrent"