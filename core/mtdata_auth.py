"""
MTDATA Authentication Module

This module handles authentication with the MTDATA API,
providing token management with automatic refresh.
"""

import logging
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class MTDataAuthClient:
    """Client for handling MTDATA API authentication."""
    
    def __init__(
        self,
        username: str,
        password: str,
        base_url: str = "https://api.mtdata.com.au",
        stack: Optional[str] = None,
        timeout: int = 30
    ) -> None:
        """
        Initialize the MTDATA authentication client.
        
        Args:
            username: MTDATA API username
            password: MTDATA API password
            base_url: Base URL for the MTDATA API
            stack: API stack to use (optional)
            timeout: Request timeout in seconds
        """
        self._username = username
        self._password = password
        self._base_url = base_url.rstrip('/')
        self._stack = stack
        self._timeout = timeout
        
        self._auth_token = None
        self._token_expiration = datetime.utcnow()
        self._token_lock = asyncio.Lock()
        
        # Session for authentication
        self._auth_session = None
        
        # Session for API calls (will be updated with stack info)
        self._api_session = None
        
        logger.debug(f"Initialized MTDATA authentication client with base URL: {base_url}")
    
    async def __aenter__(self):
        """Set up sessions for async context manager."""
        self._auth_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout)
        )
        self._api_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up sessions for async context manager."""
        if self._auth_session:
            await self._auth_session.close()
        if self._api_session:
            await self._api_session.close()
    
    async def get_auth_token(self, force_refresh: bool = False) -> str:
        """
        Get an authentication token from the MTDATA API.
        
        Args:
            force_refresh: Whether to force a token refresh regardless of expiration
            
        Returns:
            The authentication token
            
        Raises:
            Exception: If authentication fails
        """
        # If we already have a valid token and don't need to force refresh, return it
        if not force_refresh and self._auth_token and datetime.utcnow() < self._token_expiration:
            return self._auth_token
        
        # Use a semaphore to prevent multiple simultaneous token refreshes
        async with self._token_lock:
            # Check again in case another thread already refreshed the token
            if not force_refresh and self._auth_token and datetime.utcnow() < self._token_expiration:
                return self._auth_token
            
            logger.info("Acquiring new MTData API authentication token")
            
            # Initialize auth session if needed
            if not self._auth_session:
                self._auth_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                )
            
            # Prepare the request
            auth_url = f"{self._base_url}/v1/auth/login"
            payload = {
                "username": self._username,
                "password": self._password
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            try:
                # Execute the request
                async with self._auth_session.post(
                    auth_url, 
                    json=payload, 
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to get authentication token: {response.status} - {error_text}")
                        raise Exception("Failed to authenticate with MTData API")
                    
                    # Parse the response
                    auth_response = await response.json()
                    
                    if "idToken" not in auth_response or not auth_response["idToken"]:
                        logger.error("Authentication response did not contain a token")
                        raise Exception("Invalid authentication response from MTData API")
                    
                    # Store the token and set expiration time (subtract 5 minutes for safety)
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
                            logger.info(f"Using stack from authentication response: {stack}")
                    
                    # Update the client with the new base URL including the stack
                    if stack:
                        base_url_with_stack = f"{self._base_url}/{stack}"
                        
                        # Create new session with updated base URL
                        if self._api_session:
                            await self._api_session.close()
                        
                        self._api_session = aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=self._timeout)
                        )
                        
                        logger.info(f"Updated API base URL with stack: {base_url_with_stack}")
                    
                    logger.info(f"Successfully acquired MTData API token, valid until {self._token_expiration}")
                    
                    return self._auth_token
                    
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error during authentication: {str(e)}")
                raise Exception(f"Failed to connect to MTData API: {str(e)}") from e
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse authentication response: {str(e)}")
                raise Exception("Invalid response format from MTData API") from e
                
            except Exception as e:
                logger.error(f"Unexpected error during authentication: {str(e)}")
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
        
        # Logic to determine primary stack - this is a simplification
        # In a real implementation, you might need more complex logic
        
        # For now, just return the first stack
        return stacks[0]
    
    async def execute_authenticated_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None, 
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Execute an authenticated request to the MTDATA API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to base_url)
            data: Request body for POST/PUT requests
            params: Query parameters
            
        Returns:
            Response data as a dictionary
            
        Raises:
            Exception: If the request fails
        """
        # Get a valid authentication token
        token = await self.get_auth_token()
        
        # Initialize API session if needed
        if not self._api_session:
            self._api_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
        
        # Prepare the request
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            # Execute the request
            method = method.upper()
            
            if method == "GET":
                async with self._api_session.get(url, params=params, headers=headers) as response:
                    return await self._process_response(response)
                    
            elif method == "POST":
                async with self._api_session.post(url, json=data, params=params, headers=headers) as response:
                    return await self._process_response(response)
                    
            elif method == "PUT":
                async with self._api_session.put(url, json=data, params=params, headers=headers) as response:
                    return await self._process_response(response)
                    
            elif method == "DELETE":
                async with self._api_session.delete(url, params=params, headers=headers) as response:
                    return await self._process_response(response)
                    
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during API request: {str(e)}")
            raise Exception(f"Failed to connect to MTData API: {str(e)}") from e
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            raise Exception("Invalid response format from MTData API") from e
            
        except Exception as e:
            logger.error(f"Unexpected error during API request: {str(e)}")
            raise
    
    async def _process_response(self, response: aiohttp.ClientResponse) -> Dict:
        """
        Process an API response.
        
        Args:
            response: API response
            
        Returns:
            Response data as a dictionary
            
        Raises:
            Exception: If the response indicates an error
        """
        # Check for rate limiting
        if response.status == 429:
            retry_after = int(response.headers.get("Retry-After", 30))
            logger.warning(f"Rate limit exceeded. Retry after {retry_after} seconds.")
            raise Exception(f"Rate limit exceeded: Retry after {retry_after} seconds")
        
        # Check for authorization errors (may need to refresh the token)
        if response.status == 401:
            logger.warning("Unauthorized response - token may be expired")
            # Force token refresh for next request
            await self.get_auth_token(force_refresh=True)
            raise Exception("Authorization error - please retry the request")
        
        # Check for other HTTP errors
        if response.status >= 400:
            error_text = await response.text()
            logger.error(f"API error: {response.status} - {error_text}")
            raise Exception(f"API error {response.status}: {error_text}")
        
        # Parse the response
        return await response.json()


# Synchronous wrapper for compatibility with non-async code
class MTDataAuth:
    """Synchronous wrapper for MTDataAuthClient."""
    
    def __init__(
        self,
        username: str,
        password: str,
        base_url: str = "https://api.mtdata.com.au",
        stack: Optional[str] = None,
        timeout: int = 30
    ) -> None:
        """
        Initialize the MTDATA authentication client.
        
        Args:
            username: MTDATA API username
            password: MTDATA API password
            base_url: Base URL for the MTDATA API
            stack: API stack to use (optional)
            timeout: Request timeout in seconds
        """
        self._username = username
        self._password = password
        self._base_url = base_url
        self._stack = stack
        self._timeout = timeout
        
        self._auth_client = None
        self._loop = None
    
    def _ensure_async_setup(self):
        """Ensure that the async client and event loop are set up."""
        import asyncio
        
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        
        if self._auth_client is None:
            self._auth_client = MTDataAuthClient(
                username=self._username,
                password=self._password,
                base_url=self._base_url,
                stack=self._stack,
                timeout=self._timeout
            )
    
    def get_auth_token(self, force_refresh: bool = False) -> str:
        """
        Get an authentication token from the MTDATA API.
        
        Args:
            force_refresh: Whether to force a token refresh regardless of expiration
            
        Returns:
            The authentication token
        """
        self._ensure_async_setup()
        
        async def _get_token():
            async with self._auth_client:
                return await self._auth_client.get_auth_token(force_refresh)
        
        return self._loop.run_until_complete(_get_token())
    
    def execute_authenticated_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None, 
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Execute an authenticated request to the MTDATA API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to base_url)
            data: Request body for POST/PUT requests
            params: Query parameters
            
        Returns:
            Response data as a dictionary
        """
        self._ensure_async_setup()
        
        async def _execute_request():
            async with self._auth_client:
                return await self._auth_client.execute_authenticated_request(
                    method, endpoint, data, params
                )
        
        return self._loop.run_until_complete(_execute_request())