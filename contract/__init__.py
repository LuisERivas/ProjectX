"""
Contract package: shared Redis job contract for ProjectX.

Subpackages:
- contract.shared: shared schema/keys/serde/types used by gateway and worker.
- contract.worker: worker-only implementation (ContractClient, stream ops, Lua).
- contract.gateway: gateway-side Redis contract APIs.
"""

