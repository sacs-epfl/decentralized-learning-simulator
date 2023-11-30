"""
Main worker script.
"""
import asyncio
import logging
from args import get_worker_args
from multiprocessing import freeze_support

from dasklearn.worker import Worker


def run():
    args = get_worker_args()
    worker = Worker(args)
    worker.start()


if __name__ == "__main__":
    freeze_support()
    logging.basicConfig(level=logging.INFO)

    asyncio.get_event_loop().call_later(0, run)
    asyncio.get_event_loop().run_forever()
