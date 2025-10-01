# import h5py

# def h5_tree(val, pre=''):
#     items = len(val)
#     for key, val in val.items():
#         items -= 1
#         if items == 0:
#             # the last item
#             if type(val) == h5py._hl.group.Group:
#                 print(pre + '└── ' + key)
#                 h5_tree(val, pre+'    ')
#             else:
#                 try:
#                     print(pre + '└── ' + key + ' (%d)' % len(val))
#                 except TypeError:
#                     print(pre + '└── ' + key + ' (scalar)')
#         else:
#             if type(val) == h5py._hl.group.Group:
#                 print(pre + '├── ' + key)
#                 h5_tree(val, pre+'│   ')
#             else:
#                 try:
#                     print(pre + '├── ' + key + ' (%d)' % len(val))
#                 except TypeError:
#                     print(pre + '├── ' + key + ' (scalar)')

# data_path = "/overflow/dow/one_cup_rand_new_full_picks.h5"
# f = h5py.File(data_path, "r")
# h5_tree(f)
# breakpoint()

import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
import tyro
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


# ---------- Helpers ----------

def _safe_decode_image(sample) -> np.ndarray:
    """
    Return a HxWxC uint8 RGB array from an HDF5 sample which can be:
    - already a uint8 array (H, W, 3|4) or (H, W)
    - an encoded bytes blob (jpeg/png)
    """
    # If it's a scalar with bytes inside, extract item()
    if isinstance(sample, np.ndarray) and sample.ndim == 0:
        sample = sample.item()

    if isinstance(sample, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(sample))).convert("RGB")
        arr = np.array(img)
    else:
        arr = np.array(sample)
        if arr.ndim == 2:  # grayscale
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:  # RGBA -> RGB
            arr = arr[..., :3]

    if arr.dtype != np.uint8:
        # Be permissive; most vision stacks expect uint8 for "image" dtype
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _get_image_shape(ds) -> Tuple[int, int, int]:
    sample = ds[0]
    arr = _safe_decode_image(sample)
    assert arr.ndim == 3 and arr.shape[-1] == 3, f"Expected HxWx3 image, got shape {arr.shape}"
    return tuple(arr.shape)


def _read_image(ds, idx: int) -> np.ndarray:
    return _safe_decode_image(ds[idx])


def _list_episode_keys(f: h5py.File) -> List[str]:
    return sorted([k for k in f.keys() if k.startswith("episode_")])


def _infer_fps(timestamps: np.ndarray) -> float:
    """
    Try to infer FPS from timestamps of one stream.
    Supports seconds, ms, us, ns.
    Falls back to 10.0 if it can't infer.
    """
    ts = np.asarray(timestamps)
    if ts.ndim == 0 or len(ts) < 2:
        return 10.0
    ts = ts.astype("float64")
    duration = ts[-1] - ts[0]
    if duration <= 0:
        return 10.0

    # Guess time unit by magnitude of typical deltas
    dts = np.diff(ts)
    med = float(np.median(dts))
    if med <= 0:
        return 10.0

    # Heuristic thresholds
    if med > 1e7:      # ~>= 10 ms in ns
        scale = 1e9    # ns -> s
    elif med > 1e4:    # ~>= 10 ms in us
        scale = 1e6    # us -> s
    elif med > 50:     # ~>= 50 ms in ms
        scale = 1e3    # ms -> s
    else:
        scale = 1.0    # assume seconds

    dt_seconds = med / scale
    if dt_seconds <= 0:
        return 10.0
    fps = 1.0 / dt_seconds
    # bound it to something reasonable
    return float(max(1.0, min(120.0, fps)))


# ---------- Conversion ----------

@dataclass
class Args:
    input_path: Path  # path to a .h5 file or a directory containing .h5 files
    repo_name: str = "local/one_cup_rand_new_full_picks"
    wrist_serial: str = "14442C10F1EC1CD000"
    robot_type: str = "panda"     # 'panda' is appropriate for Franka/FR3
    image_writer_threads: int = 8
    image_writer_processes: int = 2
    limit_episodes: Optional[int] = None  # for quick tests


def probe_shapes_and_fps(h5_paths: List[Path], wrist_serial: str) -> Tuple[Tuple[int, int, int],
                                                                           Tuple[int, int, int],
                                                                           float,
                                                                           str]:
    """
    Peek at the first file/episode to get shapes and FPS.
    Returns: (third_shape, wrist_shape, fps, third_serial_detected)
    """
    with h5py.File(h5_paths[0], "r") as f:
        ep_keys = _list_episode_keys(f)
        if not ep_keys:
            raise RuntimeError("No 'episode_*' groups found.")

        ep0 = ep_keys[0]
        cams = f[ep0]["cams"]
        cam_keys = list(cams.keys())
        if wrist_serial not in cam_keys:
            raise RuntimeError(f"Wrist serial {wrist_serial} not found in cams: {cam_keys}")

        # choose third-person cam as any other key
        others = [k for k in cam_keys if k != wrist_serial]
        if not others:
            raise RuntimeError(f"Only one camera ({wrist_serial}) present; can't identify third-person cam.")
        third_serial = others[0]

        wrist_rgb = cams[wrist_serial]["rgb"]
        third_rgb = cams[third_serial]["rgb"]

        wrist_shape = _get_image_shape(wrist_rgb)
        third_shape = _get_image_shape(third_rgb)

        # infer fps from wrist cam timestamps if available, else joint stream
        if "timestamps" in cams[wrist_serial]:
            fps = _infer_fps(cams[wrist_serial]["timestamps"][...])
        else:
            fps = _infer_fps(f[ep0]["fr3"]["timestamps"][...])

    return third_shape, wrist_shape, fps, third_serial


def gather_h5_files(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.suffix in (".h5", ".hdf5")])
        if not files:
            raise RuntimeError(f"No .h5/.hdf5 files found in directory: {input_path}")
        return files
    else:
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        return [input_path]


def convert(args: Args) -> None:
    h5_paths = gather_h5_files(args.input_path)

    # Clean output path
    out_dir = HF_LEROBOT_HOME / args.repo_name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Probe to build features dict
    third_shape, wrist_shape, fps, probed_third_serial = probe_shapes_and_fps(h5_paths, args.wrist_serial)
    print(f"[probe] wrist shape: {wrist_shape}, third shape: {third_shape}, fps≈{fps:.3f}, "
          f"example third cam serial: {probed_third_serial}")

    features = {
        "image": {  # third-person
            "dtype": "image",
            "shape": third_shape,
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": wrist_shape,
            "names": ["height", "width", "channel"],
        },
        "state": {  # observation: x_ee (xyz + wxyz) + gripper
            "dtype": "float32",
            "shape": (8,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
        },
        "actions": {  # action: x_ee_des (xyz + wxyz) + gripper cmd
            "dtype": "float32",
            "shape": (8,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        robot_type=args.robot_type,
        fps=float(f"{fps:.3f}"),
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    # Iterate files/episodes and write frames
    for h5_path in h5_paths:
        print(f"[read] {h5_path}")
        with h5py.File(h5_path, "r") as f:
            ep_keys = _list_episode_keys(f)
            if args.limit_episodes is not None:
                ep_keys = ep_keys[: args.limit_episodes]

            for ep in ep_keys:
                g = f[ep]
                cams = g["cams"]
                cam_keys = list(cams.keys())
                if args.wrist_serial not in cam_keys:
                    print(f"[warn] Wrist serial {args.wrist_serial} missing in {ep}; skipping episode.")
                    continue
                third_candidates = [k for k in cam_keys if k != args.wrist_serial]
                if not third_candidates:
                    print(f"[warn] No third-person cam in {ep}; skipping episode.")
                    continue
                third_serial = third_candidates[0]

                wrist_rgb = cams[args.wrist_serial]["rgb"]
                third_rgb = cams[third_serial]["rgb"]

                # Streams for obs/actions
                x_ee = np.asarray(g["fr3"]["x_ee"][...])           # (N, 7)
                grip_obs = np.asarray(g["fr3"]["gripper"][...])    # (N,)
                x_ee_des = np.asarray(g["action"]["x_ee_des"][...])  # (N, 7)
                grip_act = np.asarray(g["action"]["gripper"][...])   # (N,)

                # Trim to common length
                Ns = [len(wrist_rgb), len(third_rgb), len(x_ee), len(grip_obs), len(x_ee_des), len(grip_act)]
                N = int(min(Ns))
                if N <= 0:
                    print(f"[warn] Empty episode {ep}; skipping.")
                    continue

                # Basic sanity: ensure x_ee vectors have 7 dims; flatten if needed
                x_ee = x_ee.reshape((x_ee.shape[0], -1))
                x_ee_des = x_ee_des.reshape((x_ee_des.shape[0], -1))
                assert x_ee.shape[1] == 7 and x_ee_des.shape[1] == 7, \
                    f"{ep}: expected 7D x_ee, got {x_ee.shape[1]} and {x_ee_des.shape[1]}"

                for i in range(N):
                    obs_vec = np.concatenate([x_ee[i], np.array([grip_obs[i]], dtype=np.float32)], axis=0).astype(np.float32)
                    act_vec = np.concatenate([x_ee_des[i], np.array([grip_act[i]], dtype=np.float32)], axis=0).astype(np.float32)

                    frame = {
                        "image": _read_image(third_rgb, i),
                        "wrist_image": _read_image(wrist_rgb, i),
                        "state": obs_vec,
                        "actions": act_vec,
                        "task": "<control_mode> end effector </control_mode> Pick the clear cup up and center it on a peg in the blue receptacle."  # TODO: EXPOSE THIS
                    }
                    dataset.add_frame(frame)

                dataset.save_episode()
                print(f"[write] saved episode {ep} with {N} frames")

    print(f"\nDone. Local LeRobot dataset saved at:\n  {out_dir}\n")


def main():
    args = tyro.cli(Args)
    convert(args)


if __name__ == "__main__":
    """Example command.

    uv run python scripts/convert_h5_to_lerobot.py \
        --input-path /overflow/dow/one_cup_rand_new_full_picks.h5 \
        --repo-name local/one_cup_rand_new_full_picks \
        --wrist-serial 14442C10F1EC1CD000
    """
    main()
