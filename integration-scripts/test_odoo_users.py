#!/usr/bin/env python3
"""
Corrected test for Odoo MCP with proper execute_method usage
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    """Simple MCP client for testing Odoo connection."""
    
    def __init__(self, server_path: str, env: dict = None):
        self.server_path = server_path
        self.env = env or {}
        self.process = None
        self.request_id = 0
        self.initialized = False
    
    def _get_next_id(self) -> int:
        self.request_id += 1
        return self.request_id
    
    async def start(self) -> None:
        """Start the MCP server process."""
        try:
            cmd = ["python3", self.server_path]
            logger.info(f"Starting MCP server: {' '.join(cmd)}")
            
            process_env = os.environ.copy()
            process_env.update(self.env)
            
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env
            )
            
            logger.info(f"MCP server started with PID: {self.process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request to the MCP server."""
        if not self.process:
            raise RuntimeError("MCP server not started")
        
        request_id = self._get_next_id()
        
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        
        if params:
            request["params"] = params
        
        request_json = json.dumps(request) + "\n"
        logger.info(f"Sending request: {request_json.strip()}")
        
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Read response with increased timeout
        try:
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(), 
                timeout=60.0  # Increased timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for MCP server response")
        
        if not response_line:
            raise RuntimeError("No response from MCP server")
        
        response_text = response_line.decode().strip()
        logger.info(f"Received response: {response_text}")
        
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {response_text}")
            raise RuntimeError(f"Invalid JSON response: {e}")
        
        if "error" in response:
            error = response["error"]
            raise RuntimeError(f"MCP server error: {error.get('message', 'Unknown error')}")
        
        return response.get("result", {})
    
    async def send_notification(self, method: str, params: dict = None) -> None:
        """Send a JSON-RPC notification."""
        if not self.process:
            raise RuntimeError("MCP server not started")
        
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        
        if params:
            notification["params"] = params
        
        notification_json = json.dumps(notification) + "\n"
        logger.info(f"Sending notification: {notification_json.strip()}")
        
        self.process.stdin.write(notification_json.encode())
        await self.process.stdin.drain()
    
    async def initialize(self) -> dict:
        """Initialize the MCP session."""
        logger.info("Initializing MCP session...")
        
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": False},
                "sampling": {}
            },
            "clientInfo": {
                "name": "odoo-test-client",
                "version": "1.0.0"
            }
        }
        
        result = await self.send_request("initialize", init_params)
        await self.send_notification("notifications/initialized")
        
        self.initialized = True
        logger.info("MCP session initialized successfully")
        
        return result
    
    async def list_tools(self) -> list:
        """List available tools."""
        if not self.initialized:
            await self.initialize()
        
        logger.info("Listing available tools...")
        result = await self.send_request("tools/list")
        tools = result.get("tools", [])
        logger.info(f"Found {len(tools)} tools: {[tool.get('name') for tool in tools]}")
        return tools
    
    async def call_tool(self, name: str, arguments: dict = None) -> dict:
        """Call a specific tool."""
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"Calling tool: {name} with arguments: {arguments}")
        
        params = {"name": name}
        if arguments:
            params["arguments"] = arguments
        
        result = await self.send_request("tools/call", params)
        return result
    
    async def close(self) -> None:
        """Close the MCP client."""
        if self.process:
            logger.info("Closing MCP client...")
            try:
                self.process.stdin.close()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("MCP server did not terminate gracefully, killing...")
                self.process.kill()
                await self.process.wait()
            
            logger.info("MCP client closed")
            self.process = None
            self.initialized = False


async def test_odoo_correct_method():
    """Test Odoo MCP with correct execute_method parameters."""
    script_dir = Path(__file__).parent
    odoo_server_path = script_dir / "mcp-odoo" / "run_server.py"
    
    env = {
        "ODOO_URL": "https://test.miw.group",
        "ODOO_DB": "test",
        "ODOO_USERNAME": "tomasz.kogut@miw.group",
        "ODOO_PASSWORD": os.getenv("ODOO_PASSWORD", "")
    }
    
    if not env["ODOO_PASSWORD"]:
        print("❌ ODOO_PASSWORD environment variable not set")
        print("Set it with: export ODOO_PASSWORD='your_password'")
        return
    
    client = None
    try:
        print("🔍 Testing Odoo MCP with correct method calls...")
        print("=" * 60)
        
        client = MCPClient(str(odoo_server_path), env)
        await client.start()
        await client.initialize()
        
        # Test different Odoo methods that should work
        test_methods = [
            {
                "description": "Search users with search_read method",
                "model": "res.users",
                "method": "search_read",
                "kwargs": {
                    "domain": [],
                    "fields": ["name", "login", "email", "active"],
                    "limit": 5
                }
            },
            {
                "description": "Get company information",
                "model": "res.company",
                "method": "search_read",
                "kwargs": {
                    "domain": [],
                    "fields": ["name", "email", "website"],
                    "limit": 3
                }
            },
            {
                "description": "Search user IDs only",
                "model": "res.users", 
                "method": "search",
                "kwargs": {
                    "domain": [],
                    "limit": 3
                }
            }
        ]
        
        successful_calls = 0
        
        for i, test in enumerate(test_methods, 1):
            try:
                print(f"\n🔍 Test {i}: {test['description']}")
                
                arguments = {
                    "model": test["model"],
                    "method": test["method"]
                }
                
                if "args" in test:
                    arguments["args"] = test["args"]
                if "kwargs" in test:
                    arguments["kwargs"] = test["kwargs"]
                
                print(f"📤 Calling: {arguments}")
                
                result = await client.call_tool("execute_method", arguments)
                
                print(f"✅ Success! Result:")
                
                if isinstance(result, dict) and 'content' in result:
                    for content_item in result['content']:
                        if content_item.get('type') == 'text':
                            text = content_item.get('text', '')
                            
                            # Try to parse as JSON if it looks like structured data
                            try:
                                if text.startswith('[') or text.startswith('{'):
                                    parsed = json.loads(text)
                                    print(json.dumps(parsed, indent=2, ensure_ascii=False))
                                    
                                    # Special formatting for users
                                    if test["model"] == "res.users" and isinstance(parsed, list):
                                        print(f"\n👥 **{len(parsed)} USERS FOUND:**")
                                        for user in parsed:
                                            name = user.get('name', 'N/A')
                                            login = user.get('login', 'N/A')
                                            email = user.get('email', 'N/A')
                                            active = user.get('active', 'N/A')
                                            print(f"  • {name} ({login}) - {email} [Active: {active}]")
                                    
                                    # Special formatting for companies
                                    elif test["model"] == "res.company" and isinstance(parsed, list):
                                        print(f"\n🏢 **{len(parsed)} COMPANIES FOUND:**")
                                        for company in parsed:
                                            name = company.get('name', 'N/A')
                                            email = company.get('email', 'N/A')
                                            website = company.get('website', 'N/A')
                                            print(f"  • {name} - {email} - {website}")
                                
                                else:
                                    print(text)
                            except:
                                print(text)
                
                successful_calls += 1
                
            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
                continue
        
        print(f"\n🎉 **SUMMARY:** {successful_calls}/{len(test_methods)} tests successful!")
        
        if successful_calls > 0:
            print("✅ **Odoo MCP integration is working!** You can now use this in your main business intelligence script.")
        else:
            print("❌ **No successful calls.** Check Odoo credentials and server configuration.")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if client:
            await client.close()


async def main():
    """Main function."""
    print("🚀 ODOO MCP USER RETRIEVAL TEST (CORRECTED)")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("ODOO_PASSWORD"):
        print("⚠️  Please set ODOO_PASSWORD environment variable:")
        print("   export ODOO_PASSWORD='your_password'")
        print("   python3 test_odoo_users.py")
        return 1
    
    await test_odoo_correct_method()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
