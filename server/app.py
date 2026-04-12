"""
server/app.py - re-exports from root server.py
Keeps backward compatibility if anything imports from server.app
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app, main  # noqa: F401