from typing import Any, Literal, MutableMapping

import numpy as np
import torch
from mhw.cams.oak import SingleOakCamera
from mhw.cams.zed import SingleZEDCamera
from mhw.data import DataSourceBuffer, FR3ActionSourceBuffer, FR3DataSourceBuffer
from mhw.ipc.msg import FR3EEPoseCmd, FR3State, KeyboardState, FR3JointPosCmd
from mhw.ipc.node import PartialSubscriberHealthCheckSettings, PartialTimerHealthCheckSettings

from openpi.deploy.base import AbstractDeploymentNode
# from dow_manip.policy.dow import DowContinuousDiffusionSDE
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
import time
import cv2
PSTSettingT = list[dict[str, Any]]

"""Running Notes

TODOS
-----
* implement the policy loading and inference, expose a wandb run id argument to the cli

Ideas
-----
* try to use non-temporally-uniform histories, like taking the current obs, then one from 0.1s ago, then one from 1s ago
* try to pass the gripper state as input to the policy as well. we should do this if vision alone is insufficient
    * idea 1: pass the commanded gripper state
    * idea 2: to try to get some measurement, publish a "dummy" gripper state from the FR3 node assuming a maximum
        gripper speed. The reason we need to consider this is because querying the gripper is very slow and also is
        currently blocking in the franky API.
* try to not use the third-person camera

Problems
--------
* eliminate policy jerkiness? will know how much of an issue this is after some testing

Tests
-----
* one object, fixed start, fixed goal
* one object, random start, fixed goal
* two objects, fixed starts, fixed grasp order, fixed goals
* two objects, random starts, fixed grasp order, fixed goals
* two objects, random starts, random grasp order, fixed goals
"""


class PiDemoDeploymentNode(AbstractDeploymentNode):
    """A deployment node for the Dow demo.

    The setup is as follows:
    * Robot: FR3
    * Cameras: wrist-mounted Oak 1-W camera and third-person ZED camera
    * Measurements:
        * FR3 end-effector pose history
        * camera image history
    * Policy: diffusion policy that takes in a history of end-effector poses and images and outputs a horizon of
        end-effector pose targets

    Timers:
    * plan_timer: A timer that triggers the policy planning at a specified interval.
        interval: <user-defined>
        callback: `mhw.data.deploy.dow.DowDemoDeploymentNode.plan_callback`
    * policy_timer: A timer that triggers the policy at a specified interval.
        interval: <user-defined>
        callback: `mhw.data.deploy.dow.DowDemoDeploymentNode.policy_callback`

    Subscribers:
    * fr3_state: Subscribes to the FR3 robot state.
        topic: "mhw/fr3/state"
        type: mhw.ipc.msg.FR3State
        callback: `mhw.data.record.fr3_zed.FR3ZedRecorder.fr3_state_callback`
        qos: None

    Publishers:
    * ee_pose_cmd: Publishes an end-effector pose command to the FR3 in world-frame coordinates.
        topic: "mhw/fr3/cmd/ee_pose"
        type: mhw.ipc.msg.FR3EEPoseCmd
        qos: None
    """

    # ############## #
    # INITIALIZATION #
    # ############## #

    def __init__(
        self,
        dt_policy: float,
        dt_plan: float,
        zed_settings: dict[str, Any],
        oak_settings: dict[str, Any],
        device: str = "cuda",
        fr3_buffer_len: int = 100,
        history_len: int = 5,
        dt_keyboard: float = 0.025,
        name: str = "dow_deployment_node",
        domain_id: int | Literal["any"] = "any",
        auto_start: bool = True,
        timer_health_check_settings: PartialTimerHealthCheckSettings | None = None,
        subscriber_health_check_settings: PartialSubscriberHealthCheckSettings | None = None,
    ) -> None:
        """Initialize the DowDemoDeploymentNode.

        Args:
            dt_policy: Time interval between policy executions in seconds.
            dt_plan: Time interval between policy planning in seconds.
            zed_settings: Settings for the ZED camera. Must include 'serial_number' and 'buffer_len'.
            oak_settings: Settings for the OAK camera. Must include 'serial_number' and 'buffer_len'.
            device: The device to run the policy on.
            fr3_buffer_len: Length of the FR3 data buffer.
            history_len: Length of the history to use for the policy, including current measurement. We assume
                1. the query times for the history are evenly spaced at the resolution of dt_policy
                2. the history length is the same for all data sources
            dt_keyboard: The time interval for checking keyboard state.
            name: See parent class.
            domain_id: See parent class.
            auto_start: See parent class.
            timer_health_check_settings: See parent class.
            subscriber_health_check_settings: See parent class.
        """
        # keyboard control
        self.dt_keyboard = dt_keyboard
        self._pressed_keys = set()
        self.is_running = False

        # buffers
        self.fr3_buffer_len = fr3_buffer_len
        self.history_len = history_len

        # policy
        self.dt_policy = dt_policy
        self.dt_plan = dt_plan
        self.action_buffer = []  # actions are popped off the front of the queue
        self.pi_config = _config.get_config("pi05")
        self.checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
        self.policy = _policy_config.create_trained_policy(
            self.pi_config,
            self.checkpoint_dir,
        )
        # cameras
        self.zed_camera = self._init_zed(zed_settings)
        self.oak_camera = self._init_oak(oak_settings)


        # call the constructor
        super().__init__(
            dt_policy=dt_policy,
            name=name,
            domain_id=domain_id,
            auto_start=auto_start,
            timer_health_check_settings=timer_health_check_settings,
            subscriber_health_check_settings=subscriber_health_check_settings,
        )  # self.t_zero is set in the base class constructor, all the buffers have their times zeroed to it

        # the buffers were zeroed but just in case, we also zero the camera times
        # we do this after the constructor, since that's where self.t_zero is set
        self.zed_camera.t_zero = self.t_zero
        self.oak_camera.t_zero = self.t_zero
        self.x_ee_home = np.array([0.307, 0.0, 0.487, 0.0, 1.0, 0.0, 0.0, -1.0])  # (x, y, z, qw, qx, qy, qz, gripper)
        self.action = self.x_ee_home

        # home the robot initially
        self.home_robot()

        print("Ready to start deployment. Press 'r' to start, 's' to stop and home, 'q' to quit.")

        # [DEBUG]
        # ############################################################################################
        # from hydra.utils import instantiate
        # self.dataset = instantiate(self.cfg.dataset, _convert_="all")
        # ############################################################################################

    def _init_zed(self, zed_settings: dict[str, Any]) -> SingleZEDCamera:
        """Initialize the ZED camera with the given settings."""
        assert "serial_number" in zed_settings, "ZED camera settings must include 'serial_number'"
        assert "buffer_len" in zed_settings, "ZED camera settings must include 'buffer_len'"
        return SingleZEDCamera(
            serial_number=zed_settings["serial_number"],
            buffer_len=zed_settings["buffer_len"],
            device=zed_settings.get("device", "cpu"),
            resolution=zed_settings.get("resolution", "vga"),
            fps=zed_settings.get("fps", 60),
        )

    def _init_oak(self, oak_settings: dict[str, Any]) -> SingleOakCamera:
        """Initialize the OAK camera with the given settings."""
        assert "serial_number" in oak_settings, "OAK camera settings must include 'serial_number'"
        assert "buffer_len" in oak_settings, "OAK camera settings must include 'buffer_len'"
        return SingleOakCamera(
            serial_number=oak_settings["serial_number"],
            buffer_len=oak_settings["buffer_len"],
            resolution=oak_settings.get("resolution", "vga"),
            fps=oak_settings.get("fps", 60),
        )

    def setup(self) -> tuple[PSTSettingT, PSTSettingT, PSTSettingT, dict[str, DataSourceBuffer]]:
        """Set up the deployment node.

        Specifically, creates the publisher and subscriber for the FR3 robot and creates the data buffers. The cameras
        are handled separately.
        """
        publishers = [
            {
                "name": "ee_pose_cmd",
                "topic": "mhw/fr3/cmd/ee_pose",
                "type": FR3EEPoseCmd,
                "qos": None,
            },
            {
                "name": "joint_pos_cmd",
                "topic": "mhw/fr3/cmd/joint_pos",
                "type": FR3JointPosCmd,
                "qos": None,
            }
        ]
        subscribers = [
            {
                "name": "fr3_state",
                "topic": "mhw/fr3/state",
                "type": FR3State,
                "callback": self.fr3_state_callback,
                "qos": None,
            },
            {
                "name": "keyboard_state",
                "topic": "mhw/keyboard/state",
                "type": KeyboardState,
                "callback": self.keyboard_state_callback,
                "qos": None,
            },
        ]
        timers = [
            {
                "name": "plan_timer",
                "interval": self.dt_plan,
                "callback": self.plan_callback,
            },
            {
                "name": "keyboard_timer",
                "interval": self.dt_keyboard,
                "callback": self.keyboard_timer_callback,
            },
        ]
        buffers: MutableMapping[str, DataSourceBuffer] = {
            "fr3": FR3DataSourceBuffer(buffer_len=self.fr3_buffer_len),
            "zed": self.zed_camera.buffer,
            "oak": self.oak_camera.buffer,
            "action_history": FR3ActionSourceBuffer(buffer_len=self.fr3_buffer_len),
        }
        return publishers, subscribers, timers, buffers

    def start(self) -> None:
        """Start the cameras."""
        super().start()
        self.zed_camera.start()
        self.oak_camera.start()

    # ######### #
    # CALLBACKS #
    # ######### #

    def fr3_state_callback(self, msg: FR3State) -> None:
        """Callback for the FR3 state subscriber."""
        new_data = {}
        new_data["timestamps"] = np.array(self.t_session)
        new_data["q"] = np.array(msg.q)
        new_data["dq"] = np.array(msg.dq)
        new_data["x_ee"] = np.array(msg.x_ee)
        new_data["v_ee"] = np.array(msg.v_ee)
        new_data["gripper"] = np.array(msg.gripper)
        self.buffers["fr3"].update(**new_data)

    def keyboard_state_callback(self, msg: KeyboardState) -> None:
        """Callback for keyboard state updates."""
        self._pressed_keys |= set(msg.keys)  # type: ignore  # accumulate pressed keys

    def keyboard_timer_callback(self) -> None:
        """Timer callback for handling keyboard input."""
        # handle quit request
        if "q" in self._pressed_keys:
            self._pressed_keys.remove("q")
            print("Quitting deployment node...")
            self.stop_deployment()
            self.stop()
            return

        # handle start request
        if "r" in self._pressed_keys and not self.is_running:
            self._pressed_keys.remove("r")
            self.start_deployment()

        # handle stop and home request
        if "s" in self._pressed_keys and self.is_running:
            self._pressed_keys.remove("s")
            self.stop_deployment()

    def start_deployment(self) -> None:
        """Start the deployment."""
        if self.is_running:
            return
        print("Starting deployment...")
        self.is_running = True
        self.add_action_buffer()

    def stop_deployment(self) -> None:
        """Stop the deployment and home the robot."""
        if not self.is_running:
            return
        print("Stopping deployment and homing robot...")
        self.is_running = False
        self.home_robot()

        # Reset state variables to initial values
        self._pressed_keys = set()
        self.action_buffer = []

        # Reset all buffers
        self.t_zero = self.t
        for buffer_name in self.buffers:
            self.buffers[buffer_name].clear()
            self.buffers[buffer_name].zero_time(t=self.t_zero)  # reset timestamps to current time

        self.synchronized_buffer.clear()

        # Update time zero point to ensure fresh timing
        self.zed_camera.t_zero = self.t_zero
        self.oak_camera.t_zero = self.t_zero

    def home_robot(self) -> None:
        """Home the robot to the default position."""
        self.publishers["ee_pose_cmd"].write(
            FR3EEPoseCmd(
                t=self.t,
                x_ee_des=self.x_ee_home[:7],  # type: ignore
                absolute=True,  # force homing to be absolute
                gripper=int(self.x_ee_home[7]),
                asynchronous=False,  # force homing to be blocking
            )
        )
        self.action = self.x_ee_home

    def add_action_buffer(self) -> None:
        """Add the current action to the action history buffer."""
        new_action_history = {}
        new_action_history["timestamps"] = np.array(self.t_session)
        new_action_history["x_ee_des"] = (
            np.array(self.action[:7]) if self.action is not None else np.array(self.x_ee_home[:7])
        )
        new_action_history["gripper"] = (
            np.array(self.action[7]) if self.action is not None else np.array(self.x_ee_home[7])
        )
        self.buffers["action_history"].update(**new_action_history)

    # ###### #
    # POLICY #
    # ###### #

    def plan_callback(self) -> None:
        """Callback for the policy planning timer."""
        if not self.is_running:
            return

        # 1. get the relevant history from the synchronized buffers
        # note that if the query times go too far back, we will just get the oldest data available
        # [DEBUG] if we don't subtract dt_policy here, the current measurement might be duplicated
        t_curr = self.t_session
        # query_times = t_curr - np.arange(self.history_len)[::-1] * self.dt_policy - self.dt_policy  # [DEBUG]
        query_times = t_curr - np.arange(self.history_len)[::-1] * self.dt_policy
        all_history = self.synchronized_buffer.query(query_times)

        q_history = all_history["fr3"]["q"][0]  # (7,)
        dq_history = all_history["fr3"]["dq"][0]  # (7,)
        x_ee_history = all_history["fr3"]["x_ee"][0]  # (7,)
        v_ee_history = all_history["fr3"]["v_ee"][0]  # (6,)
        gripper_history = all_history["fr3"]["gripper"][0]  # (1,)
        # map gripper_history from [-1, 1] to [0, 1]
        gripper_history = (gripper_history + 1) / 2
        zed_history = all_history["zed"]["bgra"][0]  # (H, W, 4)
        oak_history = all_history["oak"]["bgr"][0]  # (H, W, 3)

        # 2. format the history for the policy input
        # full_lowdim_obs = np.concatenate(
        #     [
        #         q_history,
        #         dq_history,
        #         x_ee_history,
        #         v_ee_history,
        #         gripper_history[:, None],
        #     ],
        #     axis=-1,
        # )  # (history_len, 28), this will get downselected by the processor to the true set of obs the policy uses
        action_history = all_history["action_history"]["x_ee_des"][0]  # (7,)
        # gripper_history = all_history["action_history"]["gripper"][0]  # (1,)
        # action_history_full = np.concatenate([action_history, gripper_history[:, None]], axis=-1)  # (history_len, 8)

        zed_history_rgb = zed_history[..., :3][..., ::-1]  # (H, W, 3)
        oak_history_rgb = oak_history[..., ::-1]  # (H, W, 3)
        # zed_history_rgb_c_first = np.transpose(zed_history_rgb, (0, 3, 1, 2))  # (history_len, 3, H, W)
        # oak_history_rgb_c_first = np.transpose(oak_history_rgb, (0, 3, 1, 2))  # (history_len, 3, H, W)

        # center crop to img_crop_size=(H=240, W=320)
        # TODO: expose these settings or don't center crop during raw data collection
        H, W = zed_history_rgb.shape[0], zed_history_rgb.shape[1]
        H_crop, W_crop = 224, 224
        h_start = (H - H_crop) // 2
        w_start = (W - W_crop) // 2
        zed_history_rgb = zed_history_rgb[h_start : h_start + H_crop, w_start : w_start + W_crop, :]
        oak_history_rgb = oak_history_rgb[h_start : h_start + H_crop, w_start : w_start + W_crop, :]

        command = {}
        command['observation/exterior_image_1_left'] = zed_history_rgb.astype(np.float32) / 255.0
        command['observation/wrist_image_left'] = oak_history_rgb.astype(np.float32) / 255.0
        command['observation/joint_position'] = q_history.astype(np.float32)
        command['observation/gripper_position'] = gripper_history.astype(np.float32)
        command['prompt'] = "sdfsdf"
        start_time = time.time()
        action_chunk = self.policy.infer(command)["actions"]
        end_time = time.time()
        print("Inferred action chunk:", action_chunk[0])
        print("Inference time:", end_time - start_time)
        # raw_dict = {
        #     "action": None,
        #     "lowdim": torch.tensor(
        #         full_lowdim_obs[None, ...], device=self.policy.device, dtype=torch.float32
        #     ),  # (1, history_len, 28 or 36)
        #     "image": torch.tensor(
        #         images[None, ...], device=self.policy.device, dtype=torch.float32
        #     ),  # (1, history_len, 2, 3, H, W)
        #     "action_history": torch.tensor(
        #         action_history_full[None, ...], device=self.policy.device, dtype=torch.float32
        #     ),
        # }
        # processed_dict = self.policy.processor(raw_dict)  # type: ignore
        # condition_cfg = processed_dict["condition_cfg"]

        # 3. run the policy to get the action output
        # TODO: measure the policy latency and make sure it's not egregious
        # with torch.no_grad():
        #     pass
            # prior = torch.zeros((1, self.Ta, self.action_dim), device=self.policy.device)
            # pred_norm, _ = self.policy.sample(
            #     prior,
            #     solver="ddpm",
            #     sample_steps=20,
            #     condition_cfg=condition_cfg,
            #     w_cfg=1.0,
            # )
            # actions = self.policy.processor.denormalize_action(pred_norm).detach().cpu().numpy()[0]  # type: ignore

        # normalize the quaternion elements and make the gripper command -1 and 1
        # actions[:, 3:7] /= np.linalg.norm(actions[:, 3:7], axis=-1, keepdims=True) + 1e-8
        # actions[:, -1] = np.sign(actions[:, -1])  # (Ta, 8)

        # # 4. fill the action buffer
        # self.action_buffer = list(actions)  # overwrite current buffer
        # action_chunk is ndaray of shape (horizon, 8), turn it into list and overwrite current buffer
        self.action_buffer = list(action_chunk)

    def policy_impl(self) -> None:
        """The implementation of the policy."""
        if not self.is_running:
            return

        # pop action off the queue
        if len(self.action_buffer) > 0:
            self.action = self.action_buffer.pop(0)
            x_ee_des = self.action[:7]
            gripper_des = 1 if self.action[7] > 0.2 else -1
            q_home = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
            x_ee_des_base = x_ee_des[0]
            x_ee_des = q_home
            x_ee_des[0] = x_ee_des_base
        else:
            print("Warning: action buffer is empty, sending no-op action")
            return
        

        # publish the action
        self.publishers["joint_pos_cmd"].write(
            FR3JointPosCmd(
                t=self.t_session,
                q_des=x_ee_des,
                gripper=gripper_des,
            )
        )

        # store the action history
        self.add_action_buffer()
