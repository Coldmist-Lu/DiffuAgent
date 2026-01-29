"""
Debug utilities for DiffuAgent BFCL testing.

This module provides debugging tools to track model calls, selector/editor usage,
and detailed input/output during testing.
"""

import os


class DebugLogger:
    """
    Debug logger for tracking inference flow.

    Usage:
        export DEBUG_DIFFUAGENT=1  # Enable debug mode
    """

    def __init__(self):
        self.enabled = os.getenv("DEBUG_DIFFUAGENT", "0") == "1"
        self.indent_level = 0
        self.call_id = 0

    def _print(self, message: str, color: str = None):
        """Print message with optional color and indentation."""
        indent = "  " * self.indent_level

        # ANSI color codes
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "purple": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "reset": "\033[0m",
            "bold": "\033[1m",
        }

        colored_message = message
        if color and color in colors:
            colored_message = f"{colors[color]}{message}{colors['reset']}"

        print(f"{indent}{colored_message}")

    def section(self, title: str):
        """Print a section header."""
        self._print(f"\n{'─'*80}", "cyan")
        self._print(f"  {title}", "bold")
        self._print(f"{'─'*80}\n", "cyan")

    def step(self, message: str):
        """Print a step message."""
        self._print(f"▶ {message}", "blue")

    def info(self, message: str):
        """Print an info message."""
        self._print(f"ℹ {message}", "white")

    def success(self, message: str):
        """Print a success message."""
        self._print(f"✓ {message}", "green")

    def warning(self, message: str):
        """Print a warning message."""
        self._print(f"⚠ {message}", "yellow")

    def error(self, message: str):
        """Print an error message."""
        self._print(f"✗ {message}", "red")

    def data(self, key: str, value: any, truncate: int = 200):
        """Print a key-value pair."""
        value_str = str(value)

        # Truncate long values
        if len(value_str) > truncate and truncate > 0:
            value_str = value_str[:truncate] + f"... (truncated, total {len(value_str)} chars)"

        self._print(f"  • {key}: {value_str}", "white")

    def json_data(self, key: str, value: dict or list, indent: int = 2):
        """Print JSON data with proper formatting."""
        value_str = json.dumps(value, indent=indent, ensure_ascii=False)

        # Check if too long
        if len(value_str) > 500:
            lines = value_str.split("\n")
            if len(lines) > 20:
                self._print(f"  • {key}:", "white")
                for line in lines[:20]:
                    self._print(f"    {line}")
                self._print(f"    ... ({len(lines) - 20} more lines)", "yellow")
            else:
                self._print(f"  • {key}:\n{value_str}", "white")
        else:
            self._print(f"  • {key}:\n{value_str}", "white")

    def indent(self):
        """Increase indentation."""
        self.indent_level += 1

    def dedent(self):
        """Decrease indentation."""
        self.indent_level = max(0, self.indent_level - 1)

    def call_start(self, component: str, method: str, call_id: int = None):
        """Mark the start of a method call."""
        if call_id is None:
            self.call_id += 1
            call_id = self.call_id

        self._print(f"\n[{call_id}] → {component}.{method}()", "purple")
        self.indent()
        return call_id

    def call_end(self, component: str, method: str, result: str = None):
        """Mark the end of a method call."""
        self.dedent()
        if result:
            self._print(f"[{self.call_id}] ← {component}.{method}() → {result}", "purple")
        else:
            self._print(f"[{self.call_id}] ← {component}.{method}()\n", "purple")

    def separator(self):
        """Print a separator."""
        self._print("", "white")


# Global debug logger instance
debug = DebugLogger()
