"""Same topology receipt and MTP correctness bridge for the FULL candidate."""
from betterscale.worker import Worker as Full
from parallel_worker import PartitionObserver


class Worker(PartitionObserver, Full):
    pass
