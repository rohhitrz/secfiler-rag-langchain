"""Environment-based configuration.

Owns the single typed view of every runtime knob. Nothing else in the codebase
reads `os.environ` directly — see `docs/adr/0003-environment-based-configuration.md`.
"""

from secfiler_rag.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
