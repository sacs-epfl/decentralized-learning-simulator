"""
Broker script.
"""
import asyncio
import logging
from asyncio import ensure_future

from args import get_broker_args
from multiprocessing import freeze_support

from dasklearn.broker import Broker


def run():
    args = get_broker_args()
    broker = Broker(args)
    ensure_future(broker.start())


if __name__ == "__main__":
    freeze_support()
    logging.basicConfig(level=logging.INFO)

    asyncio.get_event_loop().call_later(0, run)
    asyncio.get_event_loop().run_forever()
