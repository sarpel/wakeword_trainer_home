"""
Quick Launcher for Wakeword Training Platform
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.app import launch_app

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║       Wakeword Training Platform - Quick Launcher           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Launch with default settings
    launch_app(
        server_name="0.0.0.0",
        server_port=None,  # Auto-find port 7860-7870
        share=False,
        inbrowser=True
    )