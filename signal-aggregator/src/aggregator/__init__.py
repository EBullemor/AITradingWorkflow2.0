"""
Signal Aggregator Module

Combines signals from multiple strategy pods into coherent recommendations.

Components:
- SignalCombiner: Combines same-direction signals
- ConflictResolver: Handles opposing signals
- SignalDeduplicator: Removes duplicate signals
- SignalAggregator: Main orchestrator

Usage:
    from src.aggregator import SignalAggregator, AggregatedSignal
    
    # Create aggregator
    aggregator = SignalAggregator()
    
    # Aggregate signals from all pods
    recommendations = aggregator.aggregate(all_signals)
    
    # Format for output
    print(aggregator.format_recommendations(recommendations, "markdown"))
"""

from .signal_combiner import (
    AggregatedSignal,
    SignalCombiner,
)

from .conflict_resolver import (
    ConflictResolution,
    ConflictRecord,
    ConflictResolver,
    resolve_all_conflicts,
)

from .deduplication import (
    SignalDeduplicator,
    deduplicate_signals,
)

from .aggregator import (
    SignalAggregator,
    create_aggregator,
)


__all__ = [
    # Core classes
    "AggregatedSignal",
    "SignalCombiner",
    "ConflictResolution",
    "ConflictRecord",
    "ConflictResolver",
    "SignalDeduplicator",
    "SignalAggregator",
    
    # Factory functions
    "create_aggregator",
    
    # Convenience functions
    "resolve_all_conflicts",
    "deduplicate_signals",
]
