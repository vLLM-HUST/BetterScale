"""Native worker integration; imported only after platform initialization."""

import common
from joint_worker import JointWorker
from device_joint import DeviceExperts


class DeviceJointWorker(JointWorker):
    def attach(self):
        self.remote = DeviceExperts(self.model_body, common.LINKS, self.events)
        self._local_expert_forwards = []
        if not self.shadow:
            # Strong negative witness: independent generation must never execute
            # the locally loaded expert implementation after remote attachment.
            def forbidden_local_expert(*args, **kwargs):
                raise RuntimeError("local expert forward reached in remote-only mode")

            for layer in self.model_body.layers:
                expert = layer.mlp.experts
                self._local_expert_forwards.append((expert, expert.forward))
                expert.forward = forbidden_local_expert

    def seal(self):
        self.remote.sealed = True
        return super().seal()

    def detach(self):
        try:
            return super().detach()
        finally:
            for expert, forward in self._local_expert_forwards:
                expert.forward = forward
            self._local_expert_forwards.clear()
