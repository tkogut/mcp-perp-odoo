# Test workflow for the integration script
import subprocess
import os

def test_integration():
    question = "What were the top 5 selling products last month?"
    output_file = "test_report.md"
    script_path = os.path.join(os.path.dirname(__file__), "integrate_mcp.py")
    result = subprocess.run([
        "python3", script_path, question, "--output", output_file
    ], capture_output=True, text=True)
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    assert os.path.exists(output_file), "Report file was not created."
    with open(output_file) as f:
        content = f.read()
        assert "Business Question" in content
        assert "Perplexity MCP Response" in content
        assert "Odoo MCP Response" in content
    print("Test passed: Integration script ran and produced a report.")

if __name__ == "__main__":
    test_integration()
