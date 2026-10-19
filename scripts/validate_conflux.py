"""Compatibility entry point; prefer scripts/validate_algorithms.py."""
from validate_algorithms import run_one, plot, main

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
