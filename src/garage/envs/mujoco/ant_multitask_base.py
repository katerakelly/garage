import numpy as np

from garage.envs.mujoco.ant import AntEnv

class MultitaskAntEnv(AntEnv):
    def __init__(self, task={}, n_tasks=2, **kwargs):
        self._task = task
        self._goal = task['goal']
        super(MultitaskAntEnv, self).__init__(**kwargs)

    def set_task(self, task):
        self._task = task
        self._goal = task['goal'] # assume parameterization of task by single vector
