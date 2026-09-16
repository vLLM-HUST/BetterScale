"""Native worker integration; imported only after platform initialization."""

import common
from joint_worker import JointWorker
from device_joint import DeviceExperts


class DeviceJointWorker(JointWorker):
    def attach(self):
        self.remote = DeviceExperts(self.model_body, common.LINKS, self.events)

    def seal(self):
        self.remote.sealed = True
        return super().seal()
