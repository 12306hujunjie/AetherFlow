"""
Built-in Tools for ReAct Agents

This package provides a collection of built-in tools that demonstrate
the capabilities of the AetherFlow tool system and provide useful
functionality for ReAct agents.

Tool Categories:
- Compute: Mathematical and computational tools
- Search: Information retrieval and search capabilities
- Web: HTTP requests and web-based operations
- IO: File and data operations

All tools in this package are automatically exported when the tools
module is imported, making them immediately available for agent use.
"""

# Import all built-in tools for automatic registration
from .compute import *
from .search import *
from .web import *

# Note: Tools are automatically registered via @tool decorator when imported
# No explicit registration calls are needed
