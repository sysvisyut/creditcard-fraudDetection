import pytest
import sys

def main():
    print("Running test suite...")
    
    # We run pytest programmatic invoking tests/ dynamically.
    # The return code explicitly maps states
    # Note: Using args to capture output via a plugin or just run natively
    
    # Run pytest natively outputting perfectly to stdout
    exit_code = pytest.main(["-v", "--tb=short", "tests/"])
    
    if exit_code == 0:
        print("\n✅ All tests passed. Safe to run Prompt 12.")
    else:
        print("\n❌ Fix these before continuing:")
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
