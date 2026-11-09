"""Text-native UAI memory mechanisms; no provider, benchmark or v1 dependency."""

from .config import Mechanisms, RuntimeConfig
from .runtime import UAIMemoryRuntimeV2
from .language import (LifecycleInterpretation, QueryInterpretation, interpret_query,
                       interpret_statement, interpret_statements)
from .types import ConsolidationProposal, Query, SemanticFrame, SourceRef, Tier

__all__ = ["UAIMemoryRuntimeV2", "RuntimeConfig", "Mechanisms", "Query",
           "ConsolidationProposal", "SemanticFrame", "SourceRef", "Tier",
           "LifecycleInterpretation", "QueryInterpretation", "interpret_query",
           "interpret_statement", "interpret_statements"]
