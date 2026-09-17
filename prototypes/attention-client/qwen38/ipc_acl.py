# SPDX-License-Identifier: Apache-2.0
# Reused from owned LiveInfer file revision 1dc65a99196ac4cd4a159a1fcc86bdc85744851c.
"""CANN 9.0 IPC/D2D ABI; attach to the caller's existing device context.

Derived from the qualified dsv4-gang-prefix-copy prototype. Unlike that
standalone executable, never call aclInit, SetDevice, ResetDevice or Finalize.
Construct and use on the owning worker/context thread. Keys are capabilities:
keep them in private control messages, never receipts or exception arguments.
"""

import ctypes as C


class PrefixCopyACL:
    def __init__(self, library: str):
        self.lib = C.CDLL(library)
        p, pp, n, i = C.c_void_p, C.POINTER(C.c_void_p), C.c_size_t, C.c_int32
        signatures = {
            "aclrtMalloc": [pp, n, i],
            "aclrtFree": [p],
            "aclrtCreateStream": [pp],
            "aclrtDestroyStream": [p],
            "aclrtCreateEvent": [pp],
            "aclrtDestroyEvent": [p],
            "aclrtRecordEvent": [p, p],
            "aclrtQueryEventStatus": [p, C.POINTER(i)],
            "aclrtMemcpyAsync": [p, n, p, n, i, p],
            "aclrtDeviceGetBareTgid": [C.POINTER(i)],
            "aclrtIpcMemGetExportKey": [p, n, p, n, C.c_uint64],
            "aclrtIpcMemSetImportPid": [C.c_char_p, C.POINTER(i), n],
            "aclrtIpcMemImportByKey": [pp, C.c_char_p, C.c_uint64],
            "aclrtIpcMemClose": [C.c_char_p],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, i

    def _call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name}: ACL error {status}")

    def _create(self, name):
        handle = C.c_void_p()
        self._call(name, C.byref(handle))
        return handle.value

    def create_stream(self):
        return self._create("aclrtCreateStream")

    def create_event(self):
        return self._create("aclrtCreateEvent")

    def destroy_stream(self, stream):
        self._call("aclrtDestroyStream", stream)

    def destroy_event(self, event):
        self._call("aclrtDestroyEvent", event)

    def record_event(self, event, stream):
        self._call("aclrtRecordEvent", event, stream)

    def event_complete(self, event):
        status = C.c_int32()
        self._call("aclrtQueryEventStatus", event, C.byref(status))
        if status.value not in (0, 1):
            raise RuntimeError("aclrtQueryEventStatus: unexpected recorded status")
        return status.value == 1

    def copy(self, stream, target, source, size):
        self._call("aclrtMemcpyAsync", target, size, source, size, 3, stream)

    def copy_host(self, stream, target, source, size, *, to_host):
        if type(to_host) is not bool:
            raise TypeError("host copy direction must be explicit")
        self._call(
            "aclrtMemcpyAsync", target, size, source, size, 2 if to_host else 1, stream
        )

    def pid(self):
        pid = C.c_int32()
        self._call("aclrtDeviceGetBareTgid", C.byref(pid))
        return pid.value

    def export(self, address, size, recipient_pids: tuple[int, ...]):
        if (
            type(address) is not int
            or address <= 0
            or type(size) is not int
            or size <= 0
            or not recipient_pids
            or any(
                type(pid) is not int or not 0 < pid < 2**31 for pid in recipient_pids
            )
            or len(set(recipient_pids)) != len(recipient_pids)
        ):
            raise ValueError(
                "IPC export requires an allocation and distinct recipient PIDs"
            )
        key = C.create_string_buffer(65)
        self._call("aclrtIpcMemGetExportKey", address, size, key, len(key), 0)
        peers = (C.c_int32 * len(recipient_pids))(*recipient_pids)
        try:
            self._call("aclrtIpcMemSetImportPid", key.value, peers, len(peers))
        except Exception:
            self._call("aclrtIpcMemClose", key.value)
            raise
        return key.value

    def import_memory(self, key):
        ptr = C.c_void_p()
        self._call("aclrtIpcMemImportByKey", C.byref(ptr), key, 1)
        return ptr.value

    def close_mapping(self, key):
        self._call("aclrtIpcMemClose", key)

    def allocate_staging(self, size):
        """Own a standalone allocation, never a Torch caching-allocator slice."""
        if type(size) is not int or size <= 0 or size % (2 << 20):
            raise ValueError("IPC staging extent must be a positive multiple of 2 MiB")
        address = C.c_void_p()
        self._call("aclrtMalloc", C.byref(address), size, 0)  # HUGE_FIRST
        return address.value

    def free_staging(self, address):
        self._call("aclrtFree", address)
