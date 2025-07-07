"""
MTDATA API Client Module with Full Configuration

This module provides a client for interacting with the MTDATA API,
with configuration matching the C# implementation.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from tenacity import (RetryCallState, before_sleep_log, retry,
                      retry_if_exception_type, retry_if_result,
                      stop_after_attempt, wait_exponential)
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class MTDataApiClient:
    """Client for interacting with the MTDATA API, using YAML config file."""
    
    def __init__(
        self, 
        logger: logging.Logger,
        config_file: str = "config.yaml"
    ) -> None:
        """
        Initialize the MTDATA API client.
        
        Args:
            logger: Logger instance
            config_file: Path to YAML configuration file
        """
        self._logger = logger
        
        # Load configuration from YAML file
        config = self._load_config(config_file)
        
        # Get configuration values
        self._base_url = self._get_config_value(config, 'api.base_url', "https://api-transport.mtdata.com.au")
        
        # Get stack from environment or configuration, with default
        self._stack = os.environ.get("MTDATA_STACK") or \
                     self._get_config_value(config, 'api.stack') or \
                     "MTD-QUBELOGISTICS"  # Default to production stack
        
        # Try to get credentials from environment variables first, then fall back to configuration
        self._username = os.environ.get("MTDATA_API_USERNAME") or \
                        self._get_config_value(config, 'api.username') or \
                        ""
        
        self._password = os.environ.get("MTDATA_API_PASSWORD") or \
                        self._get_config_value(config, 'api.password') or \
                        ""
        
        # Validate credentials
        if not self._username or not self._password:
            self._logger.error("MTData API credentials not configured. Set MTDATA_USERNAME and MTDATA_PASSWORD environment variables or configure in config.yaml.")
        
        # Configure request timeouts
        request_timeout = self._get_config_value(config, 'api.request_timeout_seconds', 30)
        auth_timeout = 15  # Shorter timeout for auth requests
        
        # Create session for API requests
        self._session = requests.Session()
        
        # Create separate session for auth requests
        self._auth_session = requests.Session()
        
        # Configure retry policy
        retry_count = self._get_config_value(config, 'api.retry_count', 3)
        
        # Use urllib3's Retry for automatic retries on connection errors and specific status codes
        retry_strategy = Retry(
            total=retry_count,
            backoff_factor=2,  # Exponential backoff
            status_forcelist=[429, 408, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT"],
            respect_retry_after_header=True
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        
        # Initialize authentication state
        self._auth_token = None
        self._token_expiration = datetime.utcnow()
        self._token_lock = threading.Lock()
        
        self._logger.info(f"Initialized MTDATA API client with base URL: {self._base_url}")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return config_file.safe_load(f)
            else:
                self._logger.warning(f"Config file {config_file} not found, using defaults")
                return {}
        except Exception as e:
            self._logger.error(f"Error loading config file: {str(e)}")
            return {}
    
    def _get_config_value(self, config: Dict[str, Any], key_path: str, default=None) -> Any:
        """
        Get a value from the configuration using a dot-separated path.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to the value (e.g., 'api.base_url')
            default: Default value if the key doesn't exist
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_auth_token(self, force_refresh: bool = False) -> str:
        """
        Get an authentication token from the MTDATA API.
        
        Args:
            force_refresh: Whether to force a token refresh regardless of expiration
            
        Returns:
            The auth token
        """
        # If we already have a valid token and don't need to force refresh, return it
        if not force_refresh and self._auth_token and datetime.utcnow() < self._token_expiration:
            return self._auth_token
        
        # Use a lock to prevent multiple simultaneous token refreshes
        with self._token_lock:
            # Check again in case another thread already refreshed the token
            if not force_refresh and self._auth_token and datetime.utcnow() < self._token_expiration:
                return self._auth_token
            
            self._logger.info("Acquiring new MTData API authentication token")
            
            try:
                # Prepare the auth request
                url = f"{self._base_url}/v1/auth/login"
                payload = {
                    "username": self._username,
                    "password": self._password
                }
                headers = {
                    "Content-Type": "application/json"
                }
                
                # Execute the request
                response = self._auth_session.post(
                    url, 
                    json=payload, 
                    headers=headers, 
                    timeout=15
                )
                response.raise_for_status()
                
                # Parse the response
                auth_response = response.json()
                
                if "idToken" not in auth_response or not auth_response["idToken"]:
                    self._logger.error("Authentication response did not contain a token")
                    raise Exception("Invalid authentication response from MTData API")
                
                # Store the token and set expiration time
                self._auth_token = auth_response["idToken"]
                expiry_seconds = auth_response.get("expiry", 3600)
                
                # Apply safety margin (5 minutes)
                if expiry_seconds > 300:
                    expiry_seconds -= 300
                else:
                    expiry_seconds = expiry_seconds // 2
                
                self._token_expiration = datetime.utcnow() + timedelta(seconds=expiry_seconds)
                
                # Update the client's base URL with the selected stack
                stack = self._stack
                
                # If the auth response contains stacks, override with the selected one
                stacks = auth_response.get("stacks", [])
                if stacks:
                    # Use the configured stack if it exists in the available stacks
                    if self._stack and self._stack in stacks:
                        stack = self._stack
                    else:
                        # Otherwise use the primary stack from the auth response
                        stack = self._get_primary_stack(stacks)
                        self._logger.info(f"Using stack from authentication response: {stack}")
                
                # Update the client with the new base URL including the stack
                if stack:
                    # Check if the stack is already in the base URL to avoid duplicating it
                    if stack not in self._base_url:
                        base_url_with_stack = f"{self._base_url}/{stack}"
                    else:
                        # If stack is already in URL, keep the current base URL
                        base_url_with_stack = self._base_url
                    
                    # Update the base URL for subsequent requests
                    self._base_url = base_url_with_stack
                    
                    self._logger.info(f"Updated API base URL with stack: {base_url_with_stack}")
                
                self._logger.info(f"Successfully acquired MTData API token, valid until {self._token_expiration}")
                
                return self._auth_token
                
            except Exception as e:
                self._logger.error(f"Error during authentication: {str(e)}")
                raise
    
    def _get_primary_stack(self, stacks: List[str]) -> str:
        """
        Get the primary stack from a list of stacks.
        
        Args:
            stacks: List of available stacks
            
        Returns:
            Primary stack name or empty string if none found
        """
        if not stacks:
            return ""
        
        # Logic to determine primary stack - may need to be adjusted based on actual API behavior
        # For now, just return the first stack
        return stacks[0]
    
    def execute_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None, 
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Execute a request to the MTDATA API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to base_url)
            data: Request body for POST/PUT requests
            params: Query parameters
            
        Returns:
            Response data as a dictionary
            
        Raises:
            Exception: If the API returns an error
        """
        # First ensure we have a valid token
        token = self.get_auth_token()
        
        # Prepare the request
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            self._logger.debug(f"Making {method} request to {url}")
            if data:
                self._logger.debug(f"Request data: {json.dumps(data)}")
            
            if method.upper() == 'GET':
                response = self._session.get(url, params=params, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = self._session.post(url, json=data, params=params, headers=headers, timeout=30)
            elif method.upper() == 'PUT':
                response = self._session.put(url, json=data, params=params, headers=headers, timeout=30)
            elif method.upper() == 'DELETE':
                response = self._session.delete(url, params=params, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle unauthorized response by refreshing token and retrying
            if response.status_code == 401:
                self._logger.info("Unauthorized response received. Refreshing auth token before retry.")
                # Force token refresh
                token = self.get_auth_token(force_refresh=True)
                # Update authorization header
                headers["Authorization"] = f"Bearer {token}"
                
                # Retry the request
                if method.upper() == 'GET':
                    response = self._session.get(url, params=params, headers=headers, timeout=30)
                elif method.upper() == 'POST':
                    response = self._session.post(url, json=data, params=params, headers=headers, timeout=30)
                elif method.upper() == 'PUT':
                    response = self._session.put(url, json=data, params=params, headers=headers, timeout=30)
                elif method.upper() == 'DELETE':
                    response = self._session.delete(url, params=params, headers=headers, timeout=30)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            self._logger.debug(f"Received response from {url}")
            return response_data
                
        except requests.exceptions.HTTPError as e:
            self._logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API request failed: {str(e)}")
            
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Request error: {str(e)}")
            raise Exception(f"API request failed: {str(e)}")
            
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse JSON response: {str(e)}")
            raise Exception(f"Invalid API response: {str(e)}")
            
        except Exception as e:
            self._logger.error(f"Unexpected error: {str(e)}")
            raise
    
    # Example implementation of API methods
    
    def get_fleets(self) -> List[Dict[str, Any]]:
        """
        Get all fleets the user has access to.
        
        Returns:
            List of fleet objects
        """
        response = self.execute_request('POST', '/v1/fleets', data={})
        return response.get('fleets', [])
    
    def get_vehicles(self, fleet_id: int) -> List[Dict[str, Any]]:
        """
        Get all vehicles for a specific fleet.
        
        Args:
            fleet_id: ID of the fleet
            
        Returns:
            List of vehicle objects
        """
        data = {
            "fleetIds": [fleet_id],
            "includeInactive": False
        }
        response = self.execute_request('POST', '/v1/vehicles', data=data)
        return response.get('vehicles', [])
    
    def get_vehicle_history(
        self, 
        fleet_id: int, 
        vehicle_id: int, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get GPS history for a specific vehicle.
        
        Args:
            fleet_id: ID of the fleet
            vehicle_id: ID of the vehicle
            start_date: Start date for history
            end_date: End date for history
            
        Returns:
            List of history events
        """
        endpoint = f"/v1/vehicles/{fleet_id}/{vehicle_id}/history"
        data = {
            "utcStartDateTime": start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "utcEndDateTime": end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        }
        
        response = self.execute_request('POST', endpoint, data=data)
        return response.get('events', [])