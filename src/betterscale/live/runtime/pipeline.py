# Relocated from LiveInference 05ac15419c0e73650e687ceb9daffeb7874865f0.
# BetterScale change: imports use the owned betterscale.live namespace.
# SPDX-License-Identifier: Apache-2.0
"""Static, entry-owned numerical graph partitioning.

Use a scope around the numerical composition after explicit metadata construction.
Only graph building executes these hooks; a physical graph replay does not enter
Python. Transport implementations own device sequences and completion, not this
scope. This module does not allocate remote weights or schedule requests.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from inspect import signature
from typing import Protocol, TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


_ModuleT = TypeVar("_ModuleT", bound=torch.nn.Module)


@dataclass(frozen=True)
class PipelineEntry:
    """One owned subtree; receive names are forward arguments at a rank cut."""

    path: str
    rank: int
    receive: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'receive', tuple(self.receive))


@dataclass(frozen=True)
class PipelinePlan:
    """Ordered, nonoverlapping execution leaves of the whole root.

    Containers are absent. Each declared subtree executes exactly once; its
    children inherit ownership. Descendant aliases must have one owner color;
    shared/reentrant entry objects still require explicit call-site semantics
    and are rejected rather than guessed.
    """

    entries: tuple[PipelineEntry, ...]

    def __post_init__(self):
        object.__setattr__(self, 'entries', tuple(self.entries))

    @classmethod
    def from_coloring(
        cls,
        coloring: Mapping[str, int],
        *,
        interfaces: Mapping[str, type[torch.nn.Module] | torch.nn.Module],
    ) -> PipelinePlan:
        """Compile ordered owner colors against module-owned entry contracts.

        A cuttable module declares ``pipeline_inputs``: the complete tuple of
        forward argument names crossing its entry. No declaration means that
        entry cannot be a cut. Metadata and owner-local storage are not payload.
        Classes suffice, so this plan can drive owner-selective construction
        BEFORE any storage is allocated. Mapping order is execution order, not
        an inferred module traversal or a load-balancing policy.
        """
        layout = cls(tuple(PipelineEntry(path, rank) for path, rank in coloring.items()))
        layout._validate_layout()
        entries = []
        for index, entry in enumerate(layout.entries):
            receive = ()
            if index and entry.rank != layout.entries[index - 1].rank:
                module = interfaces.get(entry.path)
                declaration = getattr(module, 'pipeline_inputs', None)
                if (not isinstance(declaration, tuple) or not declaration
                        or any(not isinstance(name, str) or not name for name in declaration)
                        or len(set(declaration)) != len(declaration)):
                    raise ValueError(f'pipeline cut {entry.path!r} requires a pipeline_inputs contract')
                params = signature(module.forward).parameters
                if any(name not in params or name == 'self'
                       or params[name].kind in (params[name].VAR_POSITIONAL, params[name].VAR_KEYWORD)
                       for name in declaration):
                    raise ValueError('pipeline input is absent from explicit forward signature')
                receive = declaration
            entries.append(PipelineEntry(entry.path, entry.rank, receive))
        return cls(tuple(entries))

    @property
    def size(self) -> int:
        return max(entry.rank for entry in self.entries) + 1

    def _validate_layout(self) -> None:
        ranks = [entry.rank for entry in self.entries]
        if (not ranks or any(type(rank) is not int or rank < 0 for rank in ranks)
                or ranks != sorted(ranks) or set(ranks) != set(range(max(ranks) + 1))):
            raise ValueError('pipeline ranks must be contiguous and monotonic from zero')
        paths = [entry.path for entry in self.entries]
        if any(not path for path in paths) or len(set(paths)) != len(paths):
            raise ValueError('pipeline entries require unique nonempty module paths')
        if any(b.startswith(a + '.') for a in paths for b in paths if a != b):
            raise ValueError('pipeline owned subtrees must not overlap')

    def construct(self, path: str, factory: Callable[[], _ModuleT], *, rank: int) -> _ModuleT:
        """Construct one declared entry, allocating Torch storage only on its owner.

        The recipe still builds all containers and references. A remote factory
        runs under FakeTensorMode, including factories with explicit devices;
        its numerical forward remains unchanged. Factories must be shadow-safe
        and create their tensors inside this call, not return a prebuilt module.
        This selects weights and buffers, NOT StateTensor realization or loading.
        Do not load a checkpoint into remote fake parameters.
        """
        with self._construction_mode(path, rank) as remote:
            module = factory()
        if not isinstance(module, torch.nn.Module):
            raise TypeError('pipeline factory must return a module')
        tensors = (*module.parameters(), *module.buffers())
        if any(is_fake(tensor) != remote or tensor.device.type == 'meta'
               for tensor in tensors):
            raise ValueError('pipeline factory storage disagrees with entry ownership')
        return module

    def construct_tensors(self, path: str, factory: Callable[[], object], *, rank: int):
        """Allocate an entry's parent-held tensor tree without renaming weights.

        Used for explicit numerical inputs whose registration must stay on a
        composition container. The factory must create every tensor, just as a
        module factory must create its registered storage; no remote copy/load.
        """
        with self._construction_mode(path, rank) as remote:
            result = factory()
        tensors, _ = tree_flatten(result)
        if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError('pipeline storage factory requires a nonempty tensor tree')
        if any(is_fake(tensor) != remote or tensor.device.type == 'meta' for tensor in tensors):
            raise ValueError('pipeline factory storage disagrees with entry ownership')
        return result

    @contextmanager
    def _construction_mode(self, path: str, rank: int):
        self._validate_layout()
        if type(rank) is not int or not 0 <= rank < self.size:
            raise ValueError('pipeline rank is outside the plan')
        entries = [entry for entry in self.entries if entry.path == path]
        if not entries:
            raise ValueError('construction requires an exact pipeline entry path')
        remote = entries[0].rank != rank
        with ExitStack() as scope:
            if remote:
                scope.enter_context(FakeTensorMode(allow_non_fake_inputs=True))
            yield remote

    def bind(self, root: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
        self._validate_layout()
        ranks = [entry.rank for entry in self.entries]
        modules = tuple(root.get_submodule(entry.path) for entry in self.entries)
        # Entries cannot be reentrant, even through another subtree's alias.
        # Ordinary descendants may share one physical owner, never two colors.
        entry_ids = {id(module) for module in modules}
        owned_ids: dict[int, int] = {}
        for index, (entry, module) in enumerate(zip(self.entries, modules, strict=True)):
            ids = {id(child) for child in module.modules()}
            shared = owned_ids.keys() & ids
            if any(child in entry_ids or owned_ids[child] != entry.rank for child in shared):
                raise ValueError('aliased pipeline entries or cross-rank descendants require call-site semantics')
            owned_ids.update((child, entry.rank) for child in ids)
            is_cut = index > 0 and entry.rank != ranks[index - 1]
            if bool(entry.receive) != is_cut:
                raise ValueError('receive arguments must be declared exactly at rank cuts')
            if len(set(entry.receive)) != len(entry.receive):
                raise ValueError('duplicate pipeline receive argument')
            params = signature(module.forward).parameters
            if any(name not in params for name in entry.receive):
                raise ValueError('pipeline receive argument is absent from forward signature')
        return modules


class PipelineTransport(Protocol):
    """One call per complete cut bundle, never one handshake per tensor.

    The port fixes bundle schemas/stable storage and captures device-side credit
    waits, publication, pulls and sequence advancement. receive returns real local
    tensors. Source release follows ALL reads; destination reuse follows ALL local
    consumers (e.g. ordered on the same compute stream). Scope exit is NOT device
    completion and never releases transport credit.
    """

    def send(self, *, source: int, destination: int, bank: int,
             bundle: tuple[object, ...]) -> None: ...

    def receive(self, *, source: int, destination: int, bank: int,
                template: tuple[object, ...]) -> tuple[object, ...]: ...


class _OwnedOps(TorchDispatchMode):
    def __init__(self, execution: PipelineExecution):
        self.execution = execution

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        execution = self.execution
        if not execution._protocol:
            if execution._active is None:
                raise RuntimeError(f'unowned pipeline tensor action: {func}')
            if execution.plan.entries[execution._active].rank == execution.rank:
                leaves, _ = tree_flatten((args, kwargs or {}))
                if any(isinstance(x, torch.Tensor) and is_fake(x) for x in leaves):
                    raise RuntimeError('shadow tensor reached local pipeline computation')
        return func(*args, **(kwargs or {}))


class PipelineExecution:
    """Local public Torch hooks; numerical forwards and object paths unchanged.

    One instance is serial/non-reentrant. Scoped hooks also support a bound root
    graph entry (which bypasses root.__call__). Remote subtrees run fake tensor
    operations, not numerical kernels; their forwards must be shadow-safe, with
    no direct native launches or Python-side state mutation. This is an explicit
    modeling requirement, not a distributed shared-state detector.
    """

    def __init__(self, root: torch.nn.Module, plan: PipelinePlan, *, rank: int,
                 transport: PipelineTransport | None = None,
                 capacity_publisher=None):
        self._root = root
        self._state_owners = None
        self._capacity_publisher = capacity_publisher
        self._published_capacities = {}
        self.plan = plan
        self._modules = plan.bind(root)
        self._refresh_owner_paths(root)
        if type(rank) is not int or not 0 <= rank < plan.size:
            raise ValueError('pipeline rank is outside the plan')
        if plan.size > 1 and transport is None:
            raise ValueError('partitioned execution requires a transport')
        self.rank, self.transport = rank, transport
        self._running = False
        self._active = None
        self._fake_scope = None
        self._protocol = False

    def _refresh_owner_paths(self, root):
        # Registration aliases preserve checkpoint paths. Ownership follows
        # the actual module object, including its first/canonical root path.
        owners = {id(child): entry.rank
                  for entry, module in zip(self.plan.entries, self._modules, strict=True)
                  for child in module.modules()}
        self._owner_paths = {path: owners.get(id(module))
                             for path, module in root.named_modules()}

    def _owner_rank(self, path):
        return self._owner_paths.get(path)

    def physical_modules(self, root):
        """Select physical weight callbacks; composition containers stay visible."""
        if root is not self._root:
            raise ValueError('pipeline weight processing requires its exact bound root')
        self._modules = self.plan.bind(root)
        self._refresh_owner_paths(root)
        return tuple(module for path, module in root.named_modules()
                     if self._owner_rank(path) in (None, self.rank))

    def _activation_views(self, root):
        """Select physical lifecycle participants without deleting declarations."""
        if root is not self._root:
            raise ValueError('pipeline activation requires its exact bound root')
        self._modules = self.plan.bind(root)
        self._refresh_owner_paths(root)
        self._published_capacities = {}
        states = tuple(root.named_states())
        if states and root._live_runtime.device is None:
            raise ValueError('pipeline State shadow requires an explicit runtime device')
        self._state_owners = {}
        for path, state in states:
            rank = self._owner_rank(path.rpartition('.')[0])
            if rank is None:
                raise ValueError(f'unowned pipeline State: {path}')
            if state in self._state_owners:
                raise ValueError('aliased pipeline State declarations are not admitted')
            if rank != self.rank and state.is_bound:
                raise ValueError('remote pipeline State already has a physical binding')
            self._state_owners[state] = rank
        from betterscale.live.core.state_tensor import ExactStateCapacity
        local_domains = {state.domain for state, rank in self._state_owners.items()
                         if rank == self.rank}
        for state, rank in self._state_owners.items():
            exact = (state.domain is not None and isinstance(
                state.domain.capacity_requirement, ExactStateCapacity))
            if (rank != self.rank and not exact and state.domain not in local_domains
                    and self._capacity_publisher is None):
                raise ValueError('remote-only elastic domain needs admitted capacity publication')
        modules = tuple((path, module) for path, module in root.named_live_modules()
                        if self._owner_rank(path) in (None, self.rank))
        return modules, tuple((path, state) for path, state in states
                              if self._state_owners[state] == self.rank)

    def _publish_state_capacities(self):
        if self._capacity_publisher is None:
            return
        domains = {}
        for path, state in self._root.named_states():
            domains.setdefault(state.domain, []).append((path, state))
        keys = tuple(tuple(path for path, _ in lanes) for lanes in domains.values())
        counts = tuple(next((state._binding.num_blocks for _, state in lanes
                             if state.is_bound), None) for lanes in domains.values())
        published = tuple(self._capacity_publisher.publish_domain_capacities(keys, counts))
        if len(published) != len(domains):
            raise ValueError('incomplete pipeline capacity publication')
        from betterscale.live.core.state_tensor import ExactStateCapacity
        for (domain, lanes), count, local in zip(domains.items(), published, counts, strict=True):
            requirement = None if domain is None else domain.capacity_requirement
            if (type(count) is not int or count <= 0 or local is not None and count != local
                    or isinstance(requirement, ExactStateCapacity) and count != requirement.capacity):
                raise ValueError('pipeline capacity publication contradicts admitted capacity')
        self._published_capacities = dict(zip(domains, published, strict=True))

    def _shadow_state(self, state):
        from betterscale.live.core.state_tensor import ExactStateCapacity, StateDomain
        if isinstance(state, StateDomain):
            candidates = [s for s, rank in self._state_owners.items()
                          if s.domain is state and rank == self.plan.entries[self._active].rank]
            if not candidates:
                raise RuntimeError('pipeline State domain access crosses its owner boundary')
            state = candidates[0]
        if self._state_owners.get(state) != self.plan.entries[self._active].rank:
            raise RuntimeError('pipeline State access crosses its owner boundary')
        if state in self._shadow_values:
            return self._shadow_values[state]
        requirement = None if state.domain is None else state.domain.capacity_requirement
        if state.domain in self._published_capacities:
            capacity = self._published_capacities[state.domain]
        elif isinstance(requirement, ExactStateCapacity):
            capacity = requirement.capacity
        else:
            # A local contributor to the same domain already holds the common
            # admitted count. A remote-only elastic domain needs explicit shape
            # publication; never guess from another domain or a device budget.
            peers = [s for s in self._state_owners
                     if s.domain is state.domain and s.is_bound]
            if not peers:
                raise RuntimeError('remote elastic State has no admitted local domain capacity')
            capacity = peers[0]._binding.num_blocks
        value = torch.empty(state.physical_shape(capacity), dtype=state.storage_dtype,
                            device=self._root._live_runtime.device)
        self._shadow_values[state] = (capacity, value)
        return capacity, value

    @contextmanager
    def metadata_state_shapes(self):
        """Expose admitted State shapes, never physical storage, to host metadata.

        Use outside numerical scope and after capacity publication. Host Tensor
        construction stays real; only explicit State views are fake. This lets
        every rank construct metadata without allocating or reading remote State.
        """
        from betterscale.live.core.state_shadow import shadow_state_access
        from betterscale.live.core.state_tensor import ExactStateCapacity, StateDomain
        if self._running or self._state_owners is None:
            raise RuntimeError('metadata State shapes require activated ownership outside numerical scope')
        if any(not s.is_bound for s, owner in self._state_owners.items() if owner == self.rank):
            raise RuntimeError('metadata State shapes require a bound local generation')
        values = {}
        mode = FakeTensorMode(allow_non_fake_inputs=True)

        def resolve(state):
            if isinstance(state, StateDomain):
                state = next((s for s in self._state_owners if s.domain is state), None)
            if state not in self._state_owners:
                raise RuntimeError('metadata State does not belong to this pipeline root')
            if state not in values:
                requirement = None if state.domain is None else state.domain.capacity_requirement
                if state.domain in self._published_capacities:
                    capacity = self._published_capacities[state.domain]
                elif state.is_bound:
                    capacity = state._binding.num_blocks
                elif isinstance(requirement, ExactStateCapacity):
                    capacity = requirement.capacity
                else:
                    peers = [s for s in self._state_owners if s.domain is state.domain and s.is_bound]
                    if not peers:
                        raise RuntimeError('metadata State shape has no published capacity')
                    capacity = peers[0]._binding.num_blocks
                with mode:
                    value = torch.empty(state.physical_shape(capacity), dtype=state.storage_dtype,
                                        device=self._root._live_runtime.device)
                values[state] = (capacity, value)
            return values[state]

        with shadow_state_access(resolve):
            yield

    def _enter(self, module, args, kwargs):
        if self._active is not None:
            raise RuntimeError('nested pipeline execution boundary')
        index = self._cursor
        if index >= len(self._modules) or self._modules[index] is not module:
            raise RuntimeError('execution differs from static pipeline plan')
        entry = self.plan.entries[index]
        if entry.receive:
            source = self.plan.entries[index - 1].rank
            if self.rank in (source, entry.rank):
                call = signature(module.forward).bind(*args, **kwargs)
                call.apply_defaults()
                bundle = tuple(call.arguments[name] for name in entry.receive)
                leaves, spec = tree_flatten(bundle)
                if not leaves or any(not isinstance(x, torch.Tensor) for x in leaves):
                    raise ValueError('pipeline bundle must be a nonempty tensor tree')
                self._protocol = True
                try:
                    if self.rank == source:
                        if any(is_fake(x) for x in leaves):
                            raise RuntimeError('cannot publish a shadow pipeline bundle')
                        self.transport.send(source=source, destination=entry.rank,
                                            bank=self._bank, bundle=bundle)
                    else:
                        received = self.transport.receive(
                            source=source, destination=entry.rank,
                            bank=self._bank, template=bundle)
                        actual, actual_spec = tree_flatten(received)
                        if actual_spec != spec or any(
                            not isinstance(x, torch.Tensor) or is_fake(x)
                            or x.shape != expected.shape or x.dtype != expected.dtype
                            or x.device != expected.device
                            for x, expected in zip(actual, leaves, strict=True)
                        ):
                            raise RuntimeError('pipeline receive does not match real tensor bundle')
                        for name, value in zip(entry.receive, received, strict=True):
                            call.arguments[name] = value
                        args, kwargs = call.args, call.kwargs
                finally:
                    self._protocol = False
        if entry.rank == self.rank:
            leaves, _ = tree_flatten((args, kwargs))
            if any(isinstance(x, torch.Tensor) and is_fake(x) for x in leaves):
                raise RuntimeError('shadow tensor reached local pipeline entry')
        self._active = index
        self._cursor += 1
        if entry.rank != self.rank:
            self._fake_scope = ExitStack()
            self._fake_scope.__enter__()
            self._fake_scope.enter_context(self._fake_mode)
            if self._state_owners is not None:
                from betterscale.live.core.state_shadow import shadow_state_access
                self._fake_scope.enter_context(shadow_state_access(self._shadow_state))
        return args, kwargs

    def _exit(self, module, args, kwargs, output):
        if self._active is not None and self._modules[self._active] is module:
            if self._fake_scope is not None:
                self._fake_scope.__exit__(None, None, None)
                self._fake_scope = None
            self._active = None

    @contextmanager
    def scope(self, *, bank: int) -> Iterator[None]:
        """Build one black/white numerical program; not a host replay loop."""
        if self._running:
            raise RuntimeError('pipeline execution is not reentrant')
        if type(bank) is not int or bank not in (0, 1):
            raise ValueError('pipeline execution has exactly two banks')
        self._running = True
        self._bank, self._cursor = bank, 0
        handles = []
        try:
            self._shadow_values = {}
            self._fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            for module in self._modules:
                handles.append(module.register_forward_pre_hook(self._enter, with_kwargs=True, prepend=True))
                handles.append(module.register_forward_hook(
                    self._exit, with_kwargs=True, always_call=True))
            with _OwnedOps(self):
                yield
            if self._cursor != len(self._modules):
                raise RuntimeError('execution did not visit complete pipeline plan')
        finally:
            if self._fake_scope is not None:
                self._fake_scope.__exit__(None, None, None)
                self._fake_scope = None
            self._active = None
            self._shadow_values = {}
            for handle in handles:
                handle.remove()
            self._running = False
