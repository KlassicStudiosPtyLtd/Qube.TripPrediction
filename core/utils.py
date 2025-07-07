"""
Utility Module

This module provides utility functions for the MTDATA fleet analyzer,
including configuration loading, logging setup, and data processing.
"""

import os
import yaml
import json
import logging
import logging.config
import colorlog
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.addHandler(console_handler)
    
    # Set library loggers to a higher level to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, environment variables, or defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    # Default configuration
    config = {
        "api": {
            "token": None,
            "base_url": "https://api.mtdata.com.au",
            "max_retries": 3,
            "retry_backoff": 2.0
        },
        "output": {
            "dir": "./output"
        },
        "time_window": {
            "days": 7
        },
        "parallel": {
            "workers": 1
        }
    }
    
    # Try to load from YAML file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                config = deep_update(config, yaml_config)
        except Exception as e:
            logging.warning(f"Error loading config from {config_path}: {str(e)}")
    
    # Override with environment variables
    if os.getenv("MTDATA_API_USERNAME"):
        config["api"]["username"] = os.getenv("MTDATA_API_USERNAME")
    
    if os.getenv("MTDATA_API_PASSWORD"):
        config["api"]["password"] = os.getenv("MTDATA_API_PASSWORD")
    
    if os.getenv("MTDATA_API_TOKEN"):
        config["api"]["token"] = os.getenv("MTDATA_API_TOKEN")
    
    if os.getenv("MTDATA_BASE_URL"):
        config["api"]["base_url"] = os.getenv("MTDATA_BASE_URL")
    
    if os.getenv("MTDATA_OUTPUT_DIR"):
        config["output"]["dir"] = os.getenv("MTDATA_OUTPUT_DIR")
    
    if os.getenv("MTDATA_DAYS"):
        try:
            config["time_window"]["days"] = int(os.getenv("MTDATA_DAYS"))
        except ValueError:
            pass
    
    if os.getenv("MTDATA_PARALLEL_WORKERS"):
        try:
            config["parallel"]["workers"] = int(os.getenv("MTDATA_PARALLEL_WORKERS"))
        except ValueError:
            pass
    
    return config


def deep_update(original: Dict, update: Dict) -> Dict:
    """
    Recursively update a dictionary.
    
    Args:
        original: Original dictionary
        update: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original


def ensure_dir(directory: str) -> str:
    """
    Ensure that a directory exists.
    
    Args:
        directory: Directory path
        
    Returns:
        Absolute path to the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())


def calculate_time_window(days: int) -> tuple:
    """
    Calculate the start and end times for the data collection window.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Tuple containing (start_time, end_time) as datetime objects
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    # Round to the nearest second
    start_time = start_time.replace(microsecond=0)
    end_time = end_time.replace(microsecond=0)
    
    return start_time, end_time


def save_json_data(data: Dict[str, Any], output_dir: str, filename: str) -> str:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        output_dir: Output directory
        filename: Filename
        
    Returns:
        Path to the saved file
    """
    # Ensure the output directory exists
    ensure_dir(output_dir)
    
    # Create the full file path
    file_path = os.path.join(output_dir, filename)
    
    # Save the data to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return file_path


def create_summary_csv(output_dir: str, data: List[Dict[str, Any]]) -> str:
    """
    Create a summary CSV file from the collected data.
    
    Args:
        output_dir: Output directory
        data: List of dictionaries with vehicle data
        
    Returns:
        Path to the saved CSV file
    """
    # Ensure the output directory exists
    ensure_dir(output_dir)
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    # Save the DataFrame to CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    
    return csv_path


def format_time_for_display(dt: datetime) -> str:
    """
    Format a datetime object for display.
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")