"""Explicit offline checkpoint experiment; retains every other baseline gate."""
from betterscale.models import qwen35
from betterscale.models.qwen35.seat_scheduler import LiveStateScheduler
from betterscale.qwen35_worker import Worker as BaseWorker
import model_checkpoint

# Only the scheduler path changes: same inherited schedule, plus idle utilities.
qwen35.STATE_SCHEDULER='checkpoint_entry.Scheduler'
class Scheduler(LiveStateScheduler):
 def __init__(self,*args,**kwargs):
  model_checkpoint.install_core()
  super().__init__(*args,**kwargs)

class Worker(BaseWorker):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  model_checkpoint.install_worker()
