"""Explicitly synchronized external harness integrations.

The package contains only the host-side checkout/IPC boundary and the
FinanceHarness worker shim.  The upstream source itself is never imported by
the host process and is intentionally not part of this repository.
"""

from benchmark_harness.external_harnesses.checkout import ExternalHarnessManager
from benchmark_harness.external_harnesses.ipc import (
    IPC_PROTOCOL,
    Envelope,
    IpcChannel,
    IpcProtocolError,
)

__all__ = [
    "IPC_PROTOCOL",
    "Envelope",
    "ExternalHarnessManager",
    "IpcChannel",
    "IpcProtocolError",
]
