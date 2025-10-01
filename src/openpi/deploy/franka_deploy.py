from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
import time

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = droid_policy.make_droid_example()
print("Example keys:", example.keys())
print(example['observation/exterior_image_1_left'].shape)
print(example['observation/wrist_image_left'].shape)
print(example['observation/joint_position'].shape)
print(example['observation/gripper_position'].shape)
print(example['prompt'])
result = policy.infer(example)
print("Result keys:", result.keys())
start_time = time.time()
action_chunk = policy.infer(example)["actions"]
print("Action chunk shape:", action_chunk.shape)
print(type(action_chunk))
end_time = time.time()
print("Inference time:", end_time - start_time)