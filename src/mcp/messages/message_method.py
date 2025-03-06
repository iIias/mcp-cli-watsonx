# mcp/messages/message_types/message_method.py
from enum import Enum

class MessageMethod(str, Enum):
    """Enum of available message methods in the protocol."""
    # Ping
    PING = "ping"

    # Resource methods
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_TEMPLATES_LIST = "resources/templates/list"

    # Tool methods
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"