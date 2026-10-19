import asyncio
import logging
import pickle
import socket
from urllib.parse import urlparse
from typing import Callable, Dict

import zmq
import zmq.asyncio


ctx = zmq.asyncio.Context()


class Communication:

    def __init__(self, identity: str, listen_port: int, message_callback: Callable, is_broker: bool = False):
        self.is_broker = is_broker
        self.listen_port = listen_port
        self.message_callback = message_callback
        self.identity = identity
        self.logger = logging.getLogger(self.__class__.__name__)
        self.receive_msg_task = None

        self.listen_socket = None
        self.broker_connections: Dict = {}
        self.coordinator_connection = None

        self.bytes_sent: int = 0
        self.bytes_received: int = 0

    async def receive_messages(self):
        while True:
            identity, msg = await self.listen_socket.recv_multipart()
            self.bytes_received += len(msg)
            msg = pickle.loads(msg)
            self.logger.debug(f"Received message from {identity.decode()}: {msg}")
            try:
                self.message_callback(identity.decode(), msg)
            except Exception as exc:
                self.logger.exception(exc)

    def setup_server(self):
        self.listen_socket = ctx.socket(zmq.ROUTER)
        self.listen_socket.setsockopt(zmq.IDENTITY, self.identity.encode())
        self.listen_socket.bind("tcp://*:%d" % self.listen_port)
        self.logger.info("%s listening on port %d", "Worker" if self.is_broker else "Coordinator", self.listen_port)
        self.receive_msg_task = asyncio.create_task(self.receive_messages(), name="receive_messages")

    def start(self):
        self.setup_server()

    def connect_to_coordinator(self, coordinator_address: str, advertise_host=None):
        # Connect to the coordinator
        self.coordinator_connection = ctx.socket(zmq.DEALER)
        self.coordinator_connection.setsockopt(zmq.IDENTITY, self.identity.encode())
        self.coordinator_connection.connect(coordinator_address)

        if advertise_host is None:
            address = urlparse(coordinator_address)
            # Ask the OS which IPv4 interface routes to the coordinator. UDP
            # connect sends no payload and needs no platform-specific CLI.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                route.connect((address.hostname, address.port))
                advertise_host = route.getsockname()[0]
        ip = advertise_host
        msg = pickle.dumps({"type": "hello", "address": "tcp://%s:%d" % (ip, self.listen_port)})
        self.send_message_to_coordinator(msg)

    def connect_to(self, identity: str, address: str):
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, self.identity.encode())
        sock.connect(address)
        self.broker_connections[identity] = sock

    def send_message_to_broker(self, identity: str, msg: bytes):
        if identity not in self.broker_connections:
            raise RuntimeError("Unknown identity for sending %s" % identity)

        self.broker_connections[identity].send(msg)
        self.bytes_sent += len(msg)

    def send_message_to_all_brokers(self, msg: bytes):
        for sock in self.broker_connections.values():
            sock.send(msg)
            self.bytes_sent += len(msg)

    def send_message_to_coordinator(self, msg: bytes):
        self.coordinator_connection.send(msg)
        self.bytes_sent += len(msg)
