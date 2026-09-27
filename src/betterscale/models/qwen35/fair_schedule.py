"""Three explicit grant seams in the pinned native Scheduler.schedule.

Bind only this subclass's method; never edit the installed donor or its global
Scheduler. Preserve all native allocation, async publication and State fences.
Source identity is mandatory: fail closed rather than rewrite an unknown donor.
"""

import ast
import hashlib
import inspect
import textwrap

# Filled from the qualified core source; this is a compatibility pin, not a
# heuristic search over arbitrary versions.
SCHEDULE_SHA256 = "c98b1752aa6169277ca908124bebc02522bbc5379428540c61d23998ed2a93ae"

BALANCE_SCHEDULE_SHA256 = (
    "4e3c20f5fa6f0aac027328b3d58e78720f68dc40fb74f551c4d058cc8bd4127f"
)


def bind(native, *, balance_enabled=None):
    source = textwrap.dedent(inspect.getsource(native)).strip()
    identity = hashlib.sha256(source.encode()).hexdigest()
    if identity == BALANCE_SCHEDULE_SHA256:
        if balance_enabled is not False:
            raise ValueError(
                "Prefill round robin requires Ascend balance scheduling disabled"
            )
        # This pinned wrapper's disabled branch is exactly super().schedule(...).
        # Its imported Scheduler is the original base, not the patched module name.
        return bind(native.__globals__["Scheduler"].schedule)
    if identity != SCHEDULE_SHA256:
        raise ValueError(
            "Unqualified native Scheduler.schedule for prefill round robin"
        )
    tree = ast.parse(source)
    function = tree.body[0]
    running = next(
        n
        for n in function.body
        if isinstance(n, ast.While) and "self.running" in ast.unparse(n.test)
    )
    waiting_outer = next(
        n
        for n in function.body
        if isinstance(n, ast.If) and "not preempted_reqs" in ast.unparse(n.test)
    )
    waiting = next(n for n in waiting_outer.body if isinstance(n, ast.While))
    # Both proposed grants have already passed native token/position limits.
    for loop in (running, waiting):
        index = next(
            i
            for i, n in enumerate(loop.body)
            if isinstance(n, ast.If)
            and "self.need_mamba_block_aligned_split" in ast.unparse(n.test)
        )
        loop.body[index:index] = ast.parse(
            "num_new_tokens = self._prefill_round_robin.limit(request.request_id, num_new_tokens)\n"
        ).body
    # Unselected waiting entries must not break native admission. Use its
    # existing skipped queue so each is visited at most once in this attempt.
    index = (
        next(
            i
            for i, n in enumerate(waiting.body)
            if isinstance(n, ast.If)
            and "_is_blocked_waiting_status" in ast.unparse(n.test)
        )
        + 1
    )
    waiting.body[index:index] = ast.parse("""
if self._prefill_round_robin.grants.get(request_id, 0) <= 0:
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
""").body
    ast.fix_missing_locations(tree)
    env = dict(native.__globals__)
    exec(compile(tree, inspect.getsourcefile(native), "exec"), env)
    return env[function.name]
