# BetterScale-owned common execution and State closure; no architecture-port aliases.
# SPDX-License-Identifier: Apache-2.0
"""Prototype composable live host execution around captured model graphs."""

from betterscale.live.core.error import LiveModuleError
from betterscale.live.core.live_module import LiveModule, LiveModulePhase
from betterscale.live.core.meta_tensor import (
    MetaTensor,
    MetaTensorError,
    meta_tensor_scope,
)
from betterscale.live.core.state_tensor import (
    ElasticStateCapacity,
    ExactStateCapacity,
    ScaledStateCapacity,
    StateCapacityUnit,
    SIMDStateLane,
    SIMDStatePlan,
    SIMDStateSchema,
    StateCapacityRequirement,
    StateDomain,
    StateGenerationSnapshot,
    StateTensor,
    StateTensorError,
    admit_simd_state_plan,
    compile_exact_simd_state_plan,
    compile_simd_state_plan,
    compile_simd_state_schema,
    validate_simd_state_allocation,
    validate_state_domain_allocations,
)
from betterscale.live.runtime.capacity import (
    GlooStateCapacityCoordinator,
    StateCapacityCoordinator,
    StateFitCoordinator,
    admit_state_capacity,
)
from betterscale.live.runtime.distributed import LiveRankBinding
from betterscale.live.runtime.host_state import (
    HostStateDomainSelection,
    HostStateError,
    HostStateKey,
    HostStateSelection,
    HostStateTransferHandle,
    TorchHostStateBackend,
)
from betterscale.live.runtime.ingress import GraphCallSchema, TensorTreeIngress
from betterscale.live.runtime.invocation import (
    LiveGraphContextBackend,
    LiveInvocation,
    LiveInvocationContext,
    LiveInvocationContextBackend,
    LiveInvocationPhase,
    current_live_invocation,
)
from betterscale.live.runtime.live_graph import (
    GraphIngressProjection,
    LiveGraph,
)
from betterscale.live.runtime.live_runtime import (
    LiveRuntime,
    current_architecture_binding,
    current_live_runtime,
    install_live_runtime,
    live_runtime,
    live_runtime_installed,
)
from betterscale.live.runtime.graph_memory import GraphPoolMemorySnapshot
from betterscale.live.runtime.memory import (
    DeviceMemoryObserver,
    DeviceMemorySnapshot,
    GraphMemoryAdmission,
    LiveMemoryProfile,
    MemoryPhaseProfile,
    MemoryProfileFailure,
    StateDomainMemoryProfile,
    StateLaneMemoryProfile,
    TorchDeviceMemoryObserver,
)
from betterscale.live.runtime.meta_tensor import (
    MetaTensorRealization,
    construct_meta_tensors,
)
from betterscale.live.runtime.phase import (
    LivePhase,
    current_live_phase,
    live_phase_scope,
)
from betterscale.live.runtime.shadow import (
    ShadowForwardReceipt,
    real_ingress_tensor_write,
    replay_shadow_forward,
    write_ingress_tensor_value,
)
from betterscale.live.runtime.state_backend import (
    StateBackend,
    StateDomainRealization,
    StateGenerationRealization,
    TorchStateBackend,
    validate_state_generation_realization,
)

__all__ = (
    "DeviceMemoryObserver",
    "DeviceMemorySnapshot",
    "ElasticStateCapacity",
    "ExactStateCapacity",
    "ScaledStateCapacity",
    "StateCapacityUnit",
    "GraphCallSchema",
    "GraphIngressProjection",
    "GraphMemoryAdmission",
    "GraphPoolMemorySnapshot",
    "HostStateDomainSelection",
    "HostStateError",
    "HostStateKey",
    "HostStateSelection",
    "HostStateTransferHandle",
    "LiveGraph",
    "LiveGraphContextBackend",
    "LiveInvocation",
    "LiveInvocationContext",
    "LiveInvocationContextBackend",
    "LiveInvocationPhase",
    "LiveMemoryProfile",
    "LiveModule",
    "LiveModuleError",
    "LiveModulePhase",
    "LivePhase",
    "LiveRankBinding",
    "LiveRuntime",
    "MemoryPhaseProfile",
    "MemoryProfileFailure",
    "MetaTensor",
    "MetaTensorError",
    "MetaTensorRealization",
    "SIMDStateLane",
    "SIMDStatePlan",
    "SIMDStateSchema",
    "ShadowForwardReceipt",
    "StateBackend",
    "StateCapacityCoordinator",
    "StateFitCoordinator",
    "GlooStateCapacityCoordinator",
    "StateCapacityRequirement",
    "StateDomain",
    "StateDomainMemoryProfile",
    "StateDomainRealization",
    "StateGenerationRealization",
    "StateGenerationSnapshot",
    "StateLaneMemoryProfile",
    "StateTensor",
    "StateTensorError",
    "TensorTreeIngress",
    "TorchDeviceMemoryObserver",
    "TorchHostStateBackend",
    "TorchStateBackend",
    "admit_simd_state_plan",
    "admit_state_capacity",
    "compile_exact_simd_state_plan",
    "compile_simd_state_plan",
    "compile_simd_state_schema",
    "construct_meta_tensors",
    "current_architecture_binding",
    "current_live_invocation",
    "current_live_phase",
    "current_live_runtime",
    "install_live_runtime",
    "live_phase_scope",
    "live_runtime",
    "live_runtime_installed",
    "meta_tensor_scope",
    "real_ingress_tensor_write",
    "replay_shadow_forward",
    "validate_simd_state_allocation",
    "validate_state_domain_allocations",
    "validate_state_generation_realization",
    "write_ingress_tensor_value",
)
