"""
Job Trend Analyzer Package
Phân tích xu hướng thị trường việc làm bằng n-gram + embedding + LLM Agent
"""

__version__ = "1.0.0"
__author__ = "Job Trend Analyzer Team"

from config import (
    api_config,
    model_config,
    processing_config,
    path_config,
    setup_environment,
    validate_config
)

__all__ = [
    "api_config",
    "model_config", 
    "processing_config",
    "path_config",
    "setup_environment",
    "validate_config"
]
