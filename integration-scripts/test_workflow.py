import subprocess
import sys

def run_test():
    question = "What were the top 5 products by sales last month?"
    cmd = [sys.executable, "mcp_integration.py", question, "--output", "test_report.md"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    if result.returncode == 0:
        print("Test completed. Check test_report.md for the output.")
    else:
        print("Test failed.")

if __name__ == "__main__":
    run_test()
