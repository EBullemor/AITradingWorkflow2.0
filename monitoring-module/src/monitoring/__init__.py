"""
Monitoring Module

Provides health checks, metrics collection, and alerting for the trading pipeline.

Components:
- HealthChecks: Individual health check functions
- MetricsCollector: Collects and stores operational metrics
- Alerter: Sends alerts to Slack

Usage:
    from src.monitoring import run_all_health_checks, create_alerter
    
    # Run health checks
    report = run_all_health_checks(
        pipeline_result=result,
        data_dir=Path("data/processed"),
        recommendations=recommendations
    )
    
    # Send alert if needed
    if report.has_errors or report.has_warnings:
        alerter = create_alerter()
        alerter.send_health_alert(report)
"""

from .health_checks import (
    HealthStatus,
    AlertLevel,
    HealthCheckResult,
    HealthReport,
    check_pipeline_health,
    check_data_freshness,
    check_signal_distribution,
    check_recommendation_output,
    check_api_health,
    check_disk_space,
    run_all_health_checks,
)

from .metrics_collector import (
    PipelineMetrics,
    DailyMetrics,
    MetricsCollector,
    get_metrics_collector,
)

from .alerter import (
    AlertConfig,
    SlackAlerter,
    MockAlerter,
    create_alerter,
)


__all__ = [
    # Health checks
    "HealthStatus",
    "AlertLevel",
    "HealthCheckResult",
    "HealthReport",
    "check_pipeline_health",
    "check_data_freshness",
    "check_signal_distribution",
    "check_recommendation_output",
    "check_api_health",
    "check_disk_space",
    "run_all_health_checks",
    
    # Metrics
    "PipelineMetrics",
    "DailyMetrics",
    "MetricsCollector",
    "get_metrics_collector",
    
    # Alerter
    "AlertConfig",
    "SlackAlerter",
    "MockAlerter",
    "create_alerter",
]
