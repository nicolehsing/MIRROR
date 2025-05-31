import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.components.mirror import Mirror

def main():
    """Main execution function for interactive mode."""
    print("Initializing MIRROR Architecture...")
    mirror = Mirror()
    mirror.run_interactive()

if __name__ == "__main__":
    main()