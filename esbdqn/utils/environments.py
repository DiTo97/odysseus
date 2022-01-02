import dm_env
import typing as t

from abc import abstractmethod


class ConstrainedEnvironment(dm_env.Environment):
    """
    Abstract base class for Python constrained RL environments.
    """
    @abstractmethod
    def valid_action(self, action: t.Any):
        raise NotImplementedError()
