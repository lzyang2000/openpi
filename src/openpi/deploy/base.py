from abc import ABC, abstractmethod
from typing import Any, Literal, MutableMapping

from mhw.data import DataSourceBuffer, SynchronizedDataBuffer
from mhw.ipc.node import Node, PartialSubscriberHealthCheckSettings, PartialTimerHealthCheckSettings

PSTSettingT = list[dict[str, Any]]


class AbstractDeploymentNode(Node, ABC):
    """Abstract base class for deploying a policy on a robot.

    The deployment node reads data from the robot, passes it to a policy, and then sends the policy's action commands
    back to the robot.

    Timers:
    * policy_timer: A timer that triggers the policy at a specified interval.
        interval: <user-defined>
        callback: `mhw.data.deploy.base.AbstractDeploymentNode.policy_callback`
    """

    def __init__(
        self,
        dt_policy: float,
        name: str = "deployment_node",
        domain_id: int | Literal["any"] = "any",
        auto_start: bool = True,
        timer_health_check_settings: PartialTimerHealthCheckSettings | None = None,
        subscriber_health_check_settings: PartialSubscriberHealthCheckSettings | None = None,
    ) -> None:
        """Initialize the AbstractDeploymentNode."""
        # set up custom publishers, subscribers, timers, and buffers
        publishers, subscribers, timers, self.buffers = self.setup()
        self.t_zero = self.t
        for buffer in self.buffers.values():
            buffer.zero_time(t=self.t_zero)  # reset timestamps to zero at start of session
        self.synchronized_buffer = SynchronizedDataBuffer(self.buffers)

        # create the policy timer
        timers.append(
            {
                "name": "policy_timer",
                "interval": dt_policy,
                "callback": self.policy_callback,
            }
        )

        super().__init__(
            name=name,
            publishers=publishers,
            subscribers=subscribers,
            timers=timers,
            domain_id=domain_id,
            auto_start=auto_start,
            timer_health_check_settings=timer_health_check_settings,
            subscriber_health_check_settings=subscriber_health_check_settings,
        )

    @property
    def t_session(self) -> float:
        """Get the current time in the recording session.

        NOTE: need this to zero out the time because the precision of the numpy array may be too low to handle large
        absolute times.
        """
        return self.t - self.t_zero

    @abstractmethod
    def setup(self) -> tuple[PSTSettingT, PSTSettingT, PSTSettingT, MutableMapping[str, DataSourceBuffer]]:
        """Set up the recorder.

        This method should return all publisher, subscriber, and timer settings required by the recorder. Idiomatically,
        we expect mostly subscribers and maybe some timers - publishers should be rare. If we are recording data from
        another node, we will typically subscribe to that node's data and store it in a buffer. In the case of cameras,
        we instead time the on-demand reads of images and pull them from the camera into a buffer.

        Crucially, this method should also return a dictionary of buffers, where the keys are the names of the data
        sources and the values are instances of `DataSourceBuffer`. The callbacks should be updating fields of this
        buffer as data is received.

        Finally, even though the DataSourceBuffer is a flat dictionary, we by convention enforce a hierarchical
        structure by using slashes in the keys (e.g. "cams/0", "cams/1", etc.) to indicate the source of the data.
        """

    @abstractmethod
    def policy_impl(self) -> None:
        """Implement the policy logic.

        This method should read data from the synchronized buffer, pass it to the policy, and then send the policy's
        action commands back to the robot.

        The implementation doesn't need to handle checking whether the buffers have data or updating the synchronized
        buffer - this is handled in the `policy_callback` method, which calls this method.
        """

    def policy_callback(self) -> None:
        """Callback function for the policy timer.

        This function also automatically handles updating the synchronized data buffer. Note that in the base
        implementation, it assumes that the history is temporally uniform and synchronized with the execution of the
        policy. If this behavior is not desired, then you must overwrite `policy_callback`.
        """
        # at least one buffer is empty, so we shouldn't execute the policy yet
        for buffer in self.buffers.values():
            if not len(buffer) > 0:
                return None

        # before executing the policy, update the synchronized data buffer
        new_data = {}
        t_curr = self.t_session
        for buf_name, buffer in self.buffers.items():
            new_buffer_data = buffer.query(t_curr)
            new_data[buf_name] = new_buffer_data
        self.synchronized_buffer.update(timestamp=t_curr, values=new_data)

        # execute the policy - logic concerning how to query from the synchronized data buffer is left to the user
        self.policy_impl()
