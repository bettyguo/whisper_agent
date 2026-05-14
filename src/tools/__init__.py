"""Default agent tool surface: filesystem, notes, MCP passthrough."""

from whisper_agent.tools.builtin import register_builtin_tools
from whisper_agent.tools.mcp import (
    MCPClient,
    MCPError,
    MCPRegistry,
    MCPServerConfig,
    MCPTool,
)

__all__ = [
    "MCPClient",
    "MCPError",
    "MCPRegistry",
    "MCPServerConfig",
    "MCPTool",
    "register_builtin_tools",
]
