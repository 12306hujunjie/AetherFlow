"""
Web and HTTP Tools

Provides web-based tools for ReAct agents, including HTTP requests,
URL processing, and web content analysis.
"""

import asyncio
import json
import urllib.parse
from typing import Any

from ..decorators import tool
from ..models import ToolCategory


@tool(
    name="url_analyzer",
    description="Analyze and parse URLs to extract components and validate structure",
    category=ToolCategory.WEB,
)
def url_analyzer(url: str) -> dict[str, str | bool | dict[str, str]]:
    """
    Analyze a URL and extract its components.

    Args:
        url: URL to analyze

    Returns:
        Dictionary with URL components and analysis
    """
    if not url.strip():
        raise ValueError("URL cannot be empty")

    # Parse URL
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {str(e)}") from e

    # Extract query parameters
    query_params = dict(urllib.parse.parse_qsl(parsed.query))

    # Analyze URL characteristics
    is_secure = parsed.scheme == "https"
    has_port = parsed.port is not None
    has_params = len(query_params) > 0
    has_fragment = bool(parsed.fragment)

    # Get domain levels
    if parsed.netloc:
        domain_parts = parsed.netloc.split(".")
        if len(domain_parts) > 1:
            domain = ".".join(domain_parts[-2:])  # Get main domain
            subdomain = ".".join(domain_parts[:-2]) if len(domain_parts) > 2 else None
        else:
            domain = parsed.netloc
            subdomain = None
    else:
        domain = None
        subdomain = None

    result = {
        "original_url": url,
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "query_params": query_params,
        "domain": domain,
        "subdomain": subdomain,
        "is_secure": is_secure,
        "has_port": has_port,
        "has_params": has_params,
        "has_fragment": has_fragment,
        "path_segments": [seg for seg in parsed.path.split("/") if seg],
        "url_length": len(url),
    }

    return result


@tool(
    name="mock_http_request",
    description="Simulate HTTP requests with mock responses for testing",
    category=ToolCategory.WEB,
    timeout=15.0,
)
async def mock_http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, str | int | dict[str, Any]]:
    """
    Mock HTTP request tool that simulates web requests.

    This tool is useful for testing HTTP-based workflows without
    making actual network requests.

    Args:
        url: Target URL for the request
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: Optional request headers
        data: Optional request body data
        timeout: Request timeout in seconds

    Returns:
        Mock HTTP response with status, headers, and body
    """
    # Simulate network delay based on URL
    domain = urllib.parse.urlparse(url).netloc
    delay = 0.1 + (len(domain) * 0.01)  # Simulate variable latency
    await asyncio.sleep(min(delay, 0.5))  # Cap at 500ms

    method = method.upper()

    # Validate inputs
    if not url.strip():
        raise ValueError("URL cannot be empty")

    if method not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]:
        raise ValueError(f"Unsupported HTTP method: {method}")

    # Parse URL for mock response generation
    parsed_url = urllib.parse.urlparse(url)

    # Generate mock response based on URL and method
    if "error" in parsed_url.path.lower():
        # Simulate error responses
        status_code = 404
        response_body = {
            "error": "Not Found",
            "message": "The requested resource was not found",
        }
    elif "api" in parsed_url.path.lower():
        # Simulate API responses
        status_code = 200
        if method == "POST":
            status_code = 201
            response_body = {
                "status": "created",
                "id": 12345,
                "message": "Resource created successfully",
            }
        elif method == "DELETE":
            status_code = 204
            response_body = {}
        else:
            response_body = {
                "status": "success",
                "data": {
                    "endpoint": parsed_url.path,
                    "method": method,
                    "timestamp": "2024-09-04T10:30:00Z",
                    "mock": True,
                },
                "metadata": {
                    "version": "1.0",
                    "request_id": f"req_{abs(hash(url)) % 10000}",
                },
            }
    else:
        # Simulate HTML page responses
        status_code = 200
        page_title = f"Mock Page - {parsed_url.path}"
        response_body = f"""<!DOCTYPE html>
<html>
<head>
    <title>{page_title}</title>
</head>
<body>
    <h1>Mock Web Page</h1>
    <p>This is a mock response for URL: {url}</p>
    <p>Method: {method}</p>
    <p>Generated for testing purposes.</p>
</body>
</html>"""

    # Generate mock response headers
    response_headers = {
        "content-type": "application/json"
        if isinstance(response_body, dict)
        else "text/html",
        "server": "MockServer/1.0",
        "cache-control": "no-cache",
        "x-mock-response": "true",
    }

    # Include request headers in response for testing
    if headers:
        response_headers["x-request-headers"] = json.dumps(headers)

    # Prepare final response
    mock_response = {
        "status_code": status_code,
        "headers": response_headers,
        "body": response_body
        if isinstance(response_body, str)
        else json.dumps(response_body, indent=2),
        "url": url,
        "method": method,
        "request_headers": headers or {},
        "request_data": data,
        "response_time_ms": delay * 1000,
        "mock": True,
    }

    return mock_response


@tool(
    name="json_processor",
    description="Process JSON data: validate, extract values, transform structure",
    category=ToolCategory.WEB,
)
def json_processor(
    json_data: str | dict[str, Any],
    operation: str,
    path: str | None = None,
    new_value: Any | None = None,
) -> dict[str, Any] | list[Any] | str | bool:
    """
    Process JSON data with various operations.

    Args:
        json_data: JSON data as string or dictionary
        operation: Operation to perform (validate, extract, keys, values, size, transform)
        path: JSONPath-style path for extract operation (e.g., "data.items[0].name")
        new_value: New value for set operations

    Returns:
        Result of the JSON operation
    """
    # Parse JSON if string
    if isinstance(json_data, str):
        try:
            parsed_data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}") from e
    else:
        parsed_data = json_data

    operation = operation.lower().strip()

    if operation == "validate":
        # JSON is valid if we got here
        return {
            "valid": True,
            "type": type(parsed_data).__name__,
            "size": len(str(parsed_data)),
            "structure": _analyze_json_structure(parsed_data),
        }

    elif operation == "extract":
        if not path:
            raise ValueError("Path is required for extract operation")

        try:
            result = _extract_json_path(parsed_data, path)
            return result
        except Exception as e:
            raise ValueError(f"Failed to extract path '{path}': {str(e)}") from e

    elif operation == "keys":
        if isinstance(parsed_data, dict):
            return list(parsed_data.keys())
        else:
            raise ValueError("Keys operation only works on JSON objects")

    elif operation == "values":
        if isinstance(parsed_data, dict):
            return list(parsed_data.values())
        elif isinstance(parsed_data, list):
            return parsed_data
        else:
            return [parsed_data]

    elif operation == "size":
        if isinstance(parsed_data, dict | list):
            return len(parsed_data)
        elif isinstance(parsed_data, str):
            return len(parsed_data)
        else:
            return 1

    elif operation == "transform":
        # Simple transformation: flatten nested structure
        return _flatten_json(parsed_data)

    else:
        raise ValueError(
            f"Unsupported operation: {operation}. "
            "Supported: validate, extract, keys, values, size, transform"
        )


def _analyze_json_structure(
    data: Any, max_depth: int = 3, current_depth: int = 0
) -> dict[str, Any]:
    """Analyze JSON structure recursively."""
    if current_depth >= max_depth:
        return {"type": type(data).__name__, "truncated": True}

    if isinstance(data, dict):
        return {
            "type": "object",
            "keys": len(data),
            "properties": {
                key: _analyze_json_structure(value, max_depth, current_depth + 1)
                for key, value in list(data.items())[:5]  # Limit analysis
            },
        }
    elif isinstance(data, list):
        return {
            "type": "array",
            "length": len(data),
            "items": _analyze_json_structure(data[0], max_depth, current_depth + 1)
            if data
            else None,
        }
    else:
        return {"type": type(data).__name__}


def _extract_json_path(data: Any, path: str) -> Any:
    """Simple JSONPath-style extraction."""
    parts = path.split(".")
    current = data

    for part in parts:
        if "[" in part and "]" in part:
            # Handle array indexing like "items[0]"
            key, index_part = part.split("[")
            index = int(index_part.rstrip("]"))

            if key:
                current = current[key]
            current = current[index]
        else:
            current = current[part]

    return current


def _flatten_json(data: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested JSON structure."""
    items = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict | list):
                items.extend(_flatten_json(value, new_key, sep).items())
            else:
                items.append((new_key, value))

    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_key = f"{parent_key}[{i}]" if parent_key else str(i)

            if isinstance(value, dict | list):
                items.extend(_flatten_json(value, new_key, sep).items())
            else:
                items.append((new_key, value))
    else:
        items.append((parent_key, data))

    return dict(items)
