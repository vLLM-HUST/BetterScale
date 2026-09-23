"""Diagnostic observer on the isolated, unqualified MoE FULL candidate."""
from betterscale.worker import Worker as Candidate
from dummy_worker import ProfileObserver


class Worker(ProfileObserver, Candidate):
    pass
