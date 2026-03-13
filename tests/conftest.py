import sys
from unittest.mock import MagicMock

# sounddevice requires a PortAudio shared library at import time, which is
# not available in the test environment. Mock it so unit tests can import
# src.main without a live audio device.
sys.modules["sounddevice"] = MagicMock()
