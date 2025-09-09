
from mcp.server.fastmcp import FastMCP
from typing import Any
import httpx

mcp = FastMCP("Demo")

@mcp.tool(description="Add two ints together")
def add(a: int, b: int) -> int:
    
    return a + b

@mcp.tool(description="Subtrat two ints")
def subtract(a:int, b:int) -> int:
    return a - b

@mcp.tool(description="Multiply two ints")
def multiply(a:int, b:int) -> int:
    return a * b

@mcp.tool(description = "Divid two ints")
def divide(a:int, b:int) -> float:
    return a/b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    return "Hello, {}".format(name)



if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
