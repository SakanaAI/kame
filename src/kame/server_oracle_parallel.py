"""Compatibility entrypoint for the former parallel oracle server.

The parallel LLM stream multiplexer now lives in server_oracle.py.
"""

from .server_oracle import cli


if __name__ == "__main__":
    cli()
