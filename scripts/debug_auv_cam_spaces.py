"""Authoritative check of the gym spaces the policy is actually built from.

The saved ``env.yaml`` is ambiguous (it reports the observation Box with ``dtype: f4`` and
``low_repr 0.0 / high_repr 255.0``, and the action Box appears to have ``_shape: (2,)``), so read
the live spaces instead. These are what SB3 uses to size the feature extractor and the heads.

Beyond printing spaces, this validates the *shipped* SAC yaml against the live env: it reads
``policy`` / ``features_dim`` / ``net_arch`` out of the yaml, builds that exact policy class, and
runs an actor+critic forward pass. That is what catches drift between a config's claims and what
the env actually feeds the extractor (e.g. a features_dim that never reaches the CNN).

Two observation layouts are supported:
  * a bare ``Box``            -- Direct-v3-cam (frame stack only, symmetric critic)
  * a ``Dict{image, state}``  -- Direct-v4-cam (asymmetric: actor reads image, critic reads state)

Usage:
    ./scripts/debug_auv_cam_spaces.sh --headless --num_envs 4 --task Isaac-AUV-Tracking-Direct-v4-cam

NOTE: everything runs inside ``main()`` with a ``finally`` that always closes the sim. An
unhandled exception below would otherwise skip ``simulation_app.close()`` and leave a kit process
holding the GPU indefinitely -- i.e. the script would look "hung" instead of failing.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print the live gym spaces of a camera AUV task.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-AUV-Tracking-Direct-v4-cam",
    help="Camera AUV task to probe (Direct-v3-cam, Direct-v4-cam or Direct-v5-flow).",
)
parser.add_argument(
    "--sac_cfg",
    type=str,
    default="source/isaaclab_tasks/isaaclab_tasks/direct/isaac_auv_tracking/agents/sb3_sac_v4_cam_cfg.yaml",
    help="SAC yaml whose policy / features_dim / net_arch should be validated against the live "
    "space. NOTE this default targets a v4-cam run; pass the matching yaml for another task, "
    "e.g. --sac_cfg .../agents/sb3_sac_v5_flow_cfg.yaml for Direct-v5-flow.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
# The env owns a TiledCamera; without this the camera sensor raises at sim.reset().
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# --------------------------------------------------------------------------------------------
# space helpers -- the Dict case is why these exist: a gym Dict has no .shape / .dtype, so every
# direct read of `single_obs.shape` worked only while the observation was a bare Box.
# --------------------------------------------------------------------------------------------
def torch_dtype_of(space: gym.spaces.Space) -> torch.dtype:
    """gym's ``.dtype`` is a numpy dtype; torch.zeros needs a torch dtype."""
    return torch.from_numpy(np.zeros(1, dtype=space.dtype)).dtype


def is_dict_space(space: gym.spaces.Space) -> bool:
    # NOTE: gym.spaces.Dict subclasses Mapping, **not** dict -- `isinstance(sp, dict)` is False.
    return isinstance(space, gym.spaces.Dict)


def image_space_of(space: gym.spaces.Space) -> gym.spaces.Space:
    """The entry a feature extractor consumes: ``space['image']`` for Dict, else the space itself."""
    return space["image"] if is_dict_space(space) else space


def make_obs(space: gym.spaces.Space, batch: int, device: torch.device) -> torch.Tensor | dict:
    """A probe observation of the right shape/dtype for `space`.

    Integer (image-like) entries get random values rather than zeros so the extractor's scaling
    guard is exercised on a non-degenerate input; float entries stay zero.
    """
    if is_dict_space(space):
        out = {}
        for key, sp in space.spaces.items():
            dt = torch_dtype_of(sp)
            if not dt.is_floating_point:
                out[key] = torch.randint(0, 256, (batch, *sp.shape), dtype=dt, device=device)
            else:
                out[key] = torch.zeros((batch, *sp.shape), dtype=dt, device=device)
        return out
    dt = torch_dtype_of(space)
    if not dt.is_floating_point:
        return torch.randint(0, 256, (batch, *space.shape), dtype=dt, device=device)
    return torch.zeros((batch, *space.shape), dtype=dt, device=device)


def obs_bytes(space: gym.spaces.Space) -> int:
    """Bytes per single-env observation (sums the Dict entries)."""
    if is_dict_space(space):
        return sum(int(np.prod(sp.shape)) * int(sp.dtype.itemsize) for sp in space.spaces.values())
    return int(np.prod(space.shape)) * int(space.dtype.itemsize)


def resolve_class(dotted: str) -> type:
    """'pkg.mod.Cls' -> Cls, importing as needed."""
    mod_name, _, cls_name = dotted.rpartition(".")
    return getattr(importlib.import_module(mod_name), cls_name)


def macs_and_params(module, probe, shapes: list[str] | None = None) -> tuple[int, int, list[str]]:
    """Analytic MAC count via forward hooks, plus the feature-map shapes each conv sees.

    Cheap stand-in for fvcore/thop so the encoder trade-off can be compared without another
    dependency. Counts multiply-accumulates for Conv2d/Linear only (the dominant cost).
    """
    total = [0]

    def conv_hook(mod, inp, out):
        out_elems = out.numel() // out.shape[0]
        total[0] += out_elems * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]

    def linear_hook(mod, inp, out):
        total[0] += out.numel() // out.shape[0] * mod.in_features

    shapes = [] if shapes is None else shapes
    for c in [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]:
        c.register_forward_hook(lambda m, i, o: shapes.append(f"{m.out_channels}@{o.shape[-1]}x{o.shape[-2]}"))
    handles = [
        m.register_forward_hook(conv_hook if isinstance(m, torch.nn.Conv2d) else linear_hook)
        for m in module.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    with torch.no_grad():
        module(probe)
    for h in handles:
        h.remove()
    return total[0], sum(p.numel() for p in module.parameters()), shapes


def head_params(in_dim: int, arch: list[int]) -> int:
    """Params of an SB3 MLP head taking `in_dim` and appending a scalar output layer."""
    prev, total = in_dim, 0
    for h in list(arch) + [1]:
        total += prev * h + h
        prev = h
    return total


def count_unique(module, exclude_ids: frozenset = frozenset()) -> tuple[int, set[int]]:
    """Param count that de-duplicates shared modules (the extractor is referenced by all heads)."""
    seen: set[int] = set()
    total = 0
    for p in module.parameters():
        if id(p) in exclude_ids or id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total, seen


def main() -> int:
    """Run the checks. Returns a process exit code; the caller guarantees sim shutdown."""
    # Probes run on CPU on purpose. The env lives on cuda:0, but copying extractor/policy probes
    # onto the GPU is pointless here (they are one-off shape checks) and mixing a CPU module with
    # a CUDA probe raises "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)
    # should be the same" -- which is what previously killed this script mid-run.
    dev = torch.device("cpu")

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs, use_fabric=True)

    print("\n[spaces] ---- cfg (before gym.make) ----", flush=True)
    oc = env_cfg.observation_space
    print(f"[spaces] cfg.observation_space: type={type(oc).__name__} "
          f"shape={getattr(oc, 'shape', None)} dtype={getattr(oc, 'dtype', None)}", flush=True)
    if is_dict_space(oc):
        for k, sp in oc.spaces.items():
            print(f"[spaces]     ['{k}']: shape={sp.shape} dtype={sp.dtype}", flush=True)
    ac = env_cfg.action_space
    print(f"[spaces] cfg.action_space     : shape={getattr(ac, 'shape', None)} dtype={getattr(ac, 'dtype', None)}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        raw = env.unwrapped

        print("\n[spaces] ---- live env ----", flush=True)
        for label, sp in (("env.observation_space", env.observation_space), ("env.action_space", env.action_space)):
            print(f"[spaces] {label:22s}: type={type(sp).__name__} shape={getattr(sp, 'shape', None)}", flush=True)

        # NOTE: `env.observation_space` here is IsaacLab's *batched* space, i.e. it carries a
        # leading num_envs dimension. SB3 sizes the extractor from the single-env space instead,
        # via exactly the expression below (isaaclab_rl/sb3.py::_process_spaces).
        #
        # Careful with the type test: `unwrapped.single_observation_space` is a `gym.spaces.Dict`,
        # which subclasses `Mapping`, **not** `dict`. `isinstance(sp, dict)` is therefore False.
        ss = raw.single_observation_space
        single_obs = ss["policy"] if is_dict_space(ss) else ss
        img_space = image_space_of(single_obs)
        print(f"[spaces] single_observation_space (what SB3 sees): type={type(single_obs).__name__}", flush=True)
        if is_dict_space(single_obs):
            for k, sp in single_obs.spaces.items():
                print(f"[spaces]     ['{k}']: shape={sp.shape} dtype={sp.dtype} "
                      f"(torch: {torch_dtype_of(sp)})", flush=True)
        else:
            print(f"[spaces]     shape={single_obs.shape} dtype={single_obs.dtype} "
                  f"(torch: {torch_dtype_of(single_obs)})", flush=True)
        print(f"[spaces] extractor input (image branch): shape={img_space.shape} dtype={img_space.dtype}", flush=True)

        obs, _ = env.reset()
        print("\n[spaces] ---- after reset ----", flush=True)
        print(f"[spaces] obs        type={type(obs).__name__}", flush=True)

        # `env.reset()` returns IsaacLab's batched "policy" entry wrapped in an outer dict, so the
        # observation is nested: {'policy': {'image': ..., 'state': ...}}. Descend arbitrarily.
        def flatten(o, prefix=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from flatten(v, f"{prefix}['{k}']")
            else:
                yield prefix, o

        def describe(o, label):
            finite, lo, hi = True, float("inf"), float("-inf")
            for name, v in flatten(o):
                print(f"[spaces]   {label}{name} shape={tuple(v.shape)} dtype={v.dtype}", flush=True)
                fv = v.float()
                finite = finite and bool(torch.isfinite(fv).all())
                lo, hi = min(lo, float(fv.min())), max(hi, float(fv.max()))
            return finite, lo, hi

        finite, o_min, o_max = describe(obs, "obs")
        print(f"[spaces] obs finite (no NaN/inf) = {finite}", flush=True)
        print(f"[spaces] obs min={o_min} max={o_max}", flush=True)

        # Step a few times to confirm the contract holds and the data is finite.
        for i in range(3):
            actions = torch.zeros(raw.action_space.shape, device=raw.device)
            out = env.step(actions)
            describe(out[0], f"step {i} obs")

        from isaaclab_tasks.direct.isaac_auv_tracking.agents.cam_encoder import MidCNNExtractor

        # The candidate table below is a CONVOLUTION cost table, and MidCNNExtractor indexes
        # image_space.shape[1] / shape[2]. A vector task (Direct-v6-flowvec, image shape (D,)) has
        # no such axes, so building it here would raise IndexError. Skip with an explicit note
        # rather than silently printing nothing.
        is_raster = len(img_space.shape) == 3
        print("\n[spaces] ---- encoder cost (this is what net_arch rides on) ----", flush=True)
        cand_costs: dict[int, tuple[int, int]] = {}
        if is_raster:
            # Cost of both candidates, so the shipped features_dim can be judged rather than assumed.
            for fd in (256, 512):
                m = MidCNNExtractor(single_obs, features_dim=fd).eval().to(dev)
                probe = make_obs(single_obs, 1, dev)
                macs, params, shapes = macs_and_params(m, probe)
                cand_costs[fd] = (params, macs)
                print(f"[spaces] candidate features_dim={fd:4d}: params={params:>10,}  "
                      f"MACs={macs/1e6:8.2f}M  conv_stages={shapes}", flush=True)
        else:
            print(f"[spaces] image branch is a {tuple(img_space.shape)} VECTOR, not a raster -- "
                  "the MidCNNExtractor cost table does not apply; the shipped extractor's own "
                  "parameter count is printed below.", flush=True)

        # --- validate the SHIPPED SAC config against the live space ---------------------------
        import yaml

        cfg_path = Path(args_cli.sac_cfg)
        if not cfg_path.is_file():
            raise SystemExit(f"[spaces] SAC config not found: {cfg_path}")
        sac = yaml.safe_load(cfg_path.read_text())
        pk_raw = sac.get("policy_kwargs", {})
        fe_kwargs = pk_raw.get("features_extractor_kwargs", {})
        fd_cfg = int(fe_kwargs.get("features_dim", 256))
        net_arch = pk_raw.get("net_arch", {})
        policy_dotted = sac.get("policy", "stable_baselines3.sac.policies.SACPolicy")
        print(f"\n[spaces] ---- shipped SAC cfg: {cfg_path} ----", flush=True)
        print(f"[spaces] policy                   = {policy_dotted}", flush=True)
        print(f"[spaces] features_extractor_class = {pk_raw.get('features_extractor_class')}", flush=True)
        print(f"[spaces] features_dim={fd_cfg}  net_arch={net_arch}  share_features_extractor="
              f"{pk_raw.get('share_features_extractor')}", flush=True)

        ext_cls = resolve_class(pk_raw["features_extractor_class"]) if isinstance(
            pk_raw.get("features_extractor_class"), str) else MidCNNExtractor
        ext = ext_cls(single_obs, **fe_kwargs).eval().to(dev)
        enc_params = sum(p.numel() for p in ext.parameters())
        in_label = "in_channels" if is_raster else "in_dim"
        print(f"[spaces] extractor built OK: {in_label}={img_space.shape[0]} -> "
              f"features_dim={ext.features_dim}  ({enc_params:,} params)", flush=True)

        probe = make_obs(single_obs, 2, dev)
        with torch.no_grad():
            y = ext(probe)
        shape_ok = tuple(y.shape) == (2, fd_cfg)
        print(f"[spaces] forward on probe: out shape={tuple(y.shape)} expected={(2, fd_cfg)} -> "
              f"{shape_ok}  finite={bool(torch.isfinite(y).all())}", flush=True)

        # Encoder:head ratio is why features_dim is worth matching: the CNN is small, so the trunk
        # width mostly sizes the MLP heads.
        pi_p = head_params(fd_cfg, list(net_arch.get("pi", [256, 256])))
        print(f"[spaces] head params: pi({fd_cfg}->{list(net_arch.get('pi', []))})={pi_p:,}", flush=True)
        if is_dict_space(single_obs):
            # Asymmetric critic: its input is state (+) action, NOT the image latent, so the qf
            # head does not scale with features_dim at all. Report it to keep that visible.
            st = single_obs["state"]
            qf_in = int(np.prod(st.shape)) + int(np.prod(raw.single_action_space.shape))
            qf_p = head_params(qf_in, list(net_arch.get("qf", [256, 256])))
            print(f"[spaces] head params: qf(state {tuple(st.shape)} + action -> {qf_in}) "
                  f"arch={list(net_arch.get('qf', []))} = {qf_p:,}  (x2 critics = {2 * qf_p:,})", flush=True)
        else:
            qf_p = head_params(fd_cfg, list(net_arch.get("qf", [256, 256])))
            print(f"[spaces] head params: qf={qf_p:,}  (x2 critics) = {2 * qf_p:,}", flush=True)

        # --- build the real policy from the shipped config ------------------------------------
        # The extractor alone does not prove the config is consumable: `net_arch` as a dict, the
        # string -> class resolution, share_features_extractor and the (possibly custom) critic
        # construction all only get exercised here.
        pk = dict(pk_raw)
        if isinstance(pk.get("features_extractor_class"), str):
            pk["features_extractor_class"] = ext_cls

        single_action = raw.single_action_space
        print(f"\n[spaces] single_action_space (what SB3 sees): {single_action}", flush=True)

        policy_cls = resolve_class(policy_dotted)
        print(f"[spaces] policy class resolved: {policy_cls.__name__}", flush=True)
        policy = policy_cls(
            observation_space=single_obs,
            action_space=single_action,
            lr_schedule=lambda _: 1e-4,
            **pk,
        ).to(dev)

        # NOTE: `policy.features_extractor` is None on SACPolicy (SB3 2.8) -- the extractor lives
        # on the sub-networks. The actor always owns one; a custom state-only critic does not, so
        # `policy.critic.features_extractor` must not be assumed (it raises AttributeError on the
        # asymmetric StateCritic).
        extractor = policy.actor.features_extractor
        critic_has_ext = hasattr(policy.critic, "features_extractor")
        shared = critic_has_ext and policy.critic.features_extractor is extractor
        enc_total, enc_ids = count_unique(extractor)
        actor_total, _ = count_unique(policy.actor, frozenset(enc_ids))
        critic_total, _ = count_unique(policy.critic, frozenset(enc_ids))
        target_total, _ = count_unique(policy.critic_target)
        total_unique = enc_total + actor_total + critic_total
        print(f"[spaces] policy built OK from yaml: latent_dim={extractor.features_dim} "
              f"critic_has_own_extractor={critic_has_ext} critic_shares_actor_extractor={shared}", flush=True)
        print(f"[spaces]   extractor={enc_total:,}  actor(heads)={actor_total:,}  "
              f"critic={critic_total:,}", flush=True)
        print(f"[spaces]   trained total={total_unique:,} params "
              f"({total_unique / 1e6:.3f}M) + target copy={target_total:,}", flush=True)
        # The asymmetric critic must not contain a single convolution -- that is the whole point.
        if not critic_has_ext:
            convs = [n for n, m in policy.critic.named_modules()
                     if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))]
            print(f"[spaces] critic conv/BN layers = {len(convs)}  (0 expected for a state-only critic)",
                  flush=True)

        # End-to-end forward, handing the policy a probe tensor built from the live space so SB3's
        # own preprocess_obs path is exercised rather than bypassed. What that path does depends on
        # the space: a uint8 [0, 255] entry is an image space and gets rescaled to [0, 1], while a
        # float entry in [-1, 1] (e.g. Direct-v5-flow's optical-flow stack) is passed through
        # untouched. `make_obs` gives integer entries random values so the rescale is exercised on
        # a non-degenerate input; float entries are zeros, so this probe alone cannot prove a float
        # entry is NOT rescaled -- `scripts/debug_auv_v5_flow_obs.sh` covers that case.
        obs_batch = make_obs(single_obs, 3, dev)
        with torch.no_grad():
            latent = policy.actor.extract_features(obs_batch, extractor)
            # SB3 2.8: `action_log_prob` returns (action, log_prob), not the older 3-tuple.
            actions, log_prob = policy.actor.action_log_prob(obs_batch)
            q1, q2 = policy.critic(obs_batch, actions)
        e2e_ok = (
            tuple(latent.shape) == (3, fd_cfg)
            and tuple(actions.shape) == (3, int(single_action.shape[0]))
            and bool(torch.isfinite(actions).all() and torch.isfinite(q1).all() and torch.isfinite(q2).all())
        )
        print(f"[spaces] e2e: latent={tuple(latent.shape)} actions={tuple(actions.shape)} "
              f"q1={tuple(q1.shape)} q2={tuple(q2.shape)} ok={e2e_ok}", flush=True)

        # Buffer math: features_dim must not have changed the observation dtype, or the estimate
        # is wrong.
        #
        # `buffer_size` is the TOTAL transition count, NOT per environment: SB3's DictReplayBuffer
        # allocates `np.zeros((buffer_size // num_envs, num_envs, *shape))`, so num_envs only
        # reshapes the array. (Verified against SB3 2.8 -- 50000 at n_envs=8 and n_envs=16 both
        # allocate 7.63 GiB.) An earlier version of this line multiplied by num_envs and so
        # overstated the footprint by 8-16x, which is exactly the kind of error that makes a run
        # look unaffordable or, worse, hides a real overrun.
        obs_bytes_per_step = obs_bytes(single_obs)
        buf_size = int(sac.get("buffer_size", 0))
        buf_gib = buf_size * 2 * obs_bytes_per_step / 2**30
        print(f"[spaces] replay buffer: buffer_size={buf_size} transitions total "
              f"(num_envs={args_cli.num_envs} only reshapes this array) x 2 (obs + next_obs) x "
              f"{obs_bytes_per_step} B = {buf_gib:.2f} GiB of host RAM", flush=True)

        ok = shape_ok and e2e_ok
        print(f"\n[spaces] RESULT: " + ("OK" if ok else "MISMATCH"), flush=True)
        return 0 if ok else 1
    finally:
        env.close()


if __name__ == "__main__":
    import traceback

    try:
        rc = main()
    except BaseException as exc:  # noqa: BLE001 - must surface the failure immediately
        traceback.print_exc()
        sys.stderr.flush()
        # SimulationApp.shutdown() can block for minutes after a mid-run exception, which wastes
        # a whole GPU slot and makes the script look "hung" rather than failed. Exit hard instead.
        # Same convention as scripts/debug_auv_cam_probe.py.
        os._exit(1)
    # On success, close cleanly first; `os._exit` afterwards guarantees the exit code survives
    # (SimulationApp.close() otherwise ends the process with status 0).
    simulation_app.close()
    os._exit(rc)
