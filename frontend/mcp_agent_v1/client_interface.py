# mcp_client/client_interface.py

import asyncio
from .client import MCPClient

class MCPFrontend:
    def __init__(self, server_script_path: str):
        self.client = MCPClient()
        self.server_script_path = server_script_path
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self.client.connect_to_server(server_script_path))

    def query(self, text: str) -> str:
        return self._loop.run_until_complete(self.client.process_query(text))
    
    def stream_query(self, text: str):
        async def run():
            async for event in self.client.process_query_stream(text):
                yield event
             # Bridge async generator → sync generator
        loop = self._loop
        agen = run()

        async def iterate():
            async for ev in agen:
                yield ev

        return loop.run_until_complete(iterate())  # this won’t work directly in sync context


    def close(self):
        self._loop.run_until_complete(self.client.cleanup())
        self._loop.close()
