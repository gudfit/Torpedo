"""Backward-compat shim: import LOBPreprocessor from pipeline module.

This module remains to avoid breaking external imports and tests that reference
torpedocode.data.preprocessing.LOBPreprocessor.
"""

from __future__ import annotations

from .pipeline import LOBPreprocessor  # re-export

__all__ = ["LOBPreprocessor"]
