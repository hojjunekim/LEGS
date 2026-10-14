"""
Clients for the three LEGS policy servers (psi0, pi05, gr00t).

Every client takes the same observation and returns openpi-style output:

    obs = {"image":  (H, W, 3) uint8 ego-view RGB at native resolution,
           "state":  (18,) float32, observation.state_psi0_18 layout,
           "prompt": str,
           "reset":  bool}   # True on the first query of a rollout
    client.infer(obs) -> {"actions": (T, 18) float32}

    backend  server (Psi0 repo)                              wire       port   chunk
    psi0     src/psi/deploy/psi0_serve_simple.py             HTTP /act  22085  30
    pi05     src/openpi/deploy/serve_policy.py (websocket)   websocket  9000   16
    gr00t    src/gr00t/gr00t/deploy/gr00t_serve_simple.py    HTTP /act  5556   16

The HTTP servers take psi's RequestMessage (numpy arrays as base64 dicts); the
pi05 server takes msgpack-numpy dicts keyed like its training data
("observation/image", "states", "prompt"). Images are sent unresized: each
server applies its own training-time resize/crop.

Usage:
    from utils.psi0_client import make_client
    policy = make_client("psi0", host="localhost")
    actions = policy.infer(obs)["actions"]
"""

import numpy as np
import requests
from base64 import b64encode, b64decode
from numpy.lib.format import dtype_to_descr, descr_to_dtype


BACKENDS = {
    "psi0": {"protocol": "http", "port": 22085, "chunk": 30, "image_key": "egocentric"},
    "pi05": {"protocol": "websocket", "port": 9000, "chunk": 16},
    "gr00t": {"protocol": "http", "port": 5556, "chunk": 16, "image_key": "ego_view"},
}


def _numpy_serialize(o):
    if isinstance(o, np.ndarray):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Not a numpy array: {type(o)}")


def _numpy_deserialize(dct):
    if "__numpy__" in dct:
        arr = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return arr.reshape(dct["shape"]) if dct["shape"] else arr[0]
    return dct


def _convert_numpy(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {k: _convert_numpy(v, func) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_numpy(item, func) for item in data]
    elif isinstance(data, np.ndarray):
        return func(data)
    return data


def _as_chunk(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    return actions


class HTTPPolicyClient:
    """psi RequestMessage over HTTP, for the psi0 and gr00t simple servers."""

    def __init__(self, host="localhost", port=22085, image_key="egocentric", timeout=30.0):
        self.url = f"http://{host}:{port}/act"
        self.image_key = image_key
        self.timeout = timeout
        # Stay alive on failure so the sim can start before the server does;
        # callers read `connected` instead of assuming success.
        try:
            r = requests.get(f"http://{host}:{port}/health", timeout=5)
            r.raise_for_status()
            self.connected = True
            print(f"[PolicyClient] Connected to {host}:{port}")
        except Exception as e:
            self.connected = False
            print(f"[PolicyClient] Warning: health check failed: {e}")

    def infer(self, obs):
        image = np.ascontiguousarray(obs["image"], dtype=np.uint8)
        states = np.asarray(obs["state"], dtype=np.float32)
        if states.ndim == 1:
            states = states[np.newaxis, :]  # server expects (To, Ds)
        request = {
            "image": {self.image_key: image},
            "instruction": obs.get("prompt", ""),
            # psi0 drops its cached RTC chunk when "reset" is in history.
            "history": {"reset": True} if obs.get("reset") else {},
            "state": {"states": states},
            "condition": {},
            "gt_action": [],
            "dataset_name": "",
            "timestamp": "",
        }
        response = requests.post(self.url, json=_convert_numpy(request, _numpy_serialize),
                                 timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        # Errors come back as {"status": ...} (psi0 sends it as a JSON string).
        if isinstance(result, str) or "status" in result:
            raise RuntimeError(f"[PolicyClient] Server returned error: {result}")
        result = _convert_numpy(result, _numpy_deserialize)
        return {"actions": _as_chunk(result.get("action", result.get("actions")))}


class WebsocketPolicyClient:
    """openpi msgpack-numpy websocket protocol, for the pi05 server.

    Unlike openpi_client.WebsocketClientPolicy this does not block until the
    server is up; it reconnects on the next infer() instead.
    """

    def __init__(self, host="localhost", port=9000, timeout=30.0):
        self.uri = f"ws://{host}:{port}"
        self.timeout = timeout
        self.metadata = {}
        self._ws = None
        self.connected = self._connect()

    def _connect(self):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        try:
            self._ws = connect(self.uri, compression=None, max_size=None, open_timeout=5)
            self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=self.timeout))
            print(f"[PolicyClient] Connected to {self.uri}")
            return True
        except Exception as e:
            self._ws = None
            print(f"[PolicyClient] Warning: cannot connect to {self.uri}: {e}")
            return False

    def infer(self, obs):
        from openpi_client import msgpack_numpy
        if self._ws is None and not self._connect():
            raise ConnectionError(f"[PolicyClient] No server at {self.uri}")
        request = {
            "observation/image": np.ascontiguousarray(obs["image"], dtype=np.uint8),
            "states": np.asarray(obs["state"], dtype=np.float32).reshape(-1),
            "prompt": obs.get("prompt", ""),
            "reset": bool(obs.get("reset", False)),
        }
        try:
            self._ws.send(msgpack_numpy.packb(request))
            response = self._ws.recv(timeout=self.timeout)
        except Exception:
            self._ws = None
            raise
        if isinstance(response, str):
            # The server sends the traceback as text, then closes.
            self._ws = None
            raise RuntimeError(f"[PolicyClient] Server returned error:\n{response}")
        result = msgpack_numpy.unpackb(response)
        return {"actions": _as_chunk(result["actions"]),
                "server_timing": result.get("server_timing")}


def make_client(backend, host="localhost", port=None, timeout=30.0):
    """Client for `backend` ("psi0", "pi05" or "gr00t"); port defaults per backend."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {list(BACKENDS)}")
    spec = BACKENDS[backend]
    port = port or spec["port"]
    if spec["protocol"] == "websocket":
        return WebsocketPolicyClient(host=host, port=port, timeout=timeout)
    return HTTPPolicyClient(host=host, port=port, image_key=spec["image_key"], timeout=timeout)
