from mcp.server.fastmcp import FastMCP
from typing import Any
import httpx

mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool()
def subtract(a:int, b:int) -> int:
    return a - b

@mcp.tool()
def multiply(a:int, b:int) -> int:
    return a * b

@mcp.tool()
def divide(a:int, b:int) -> float:
    return a/b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    return "Hello, {}".format(name)



if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')