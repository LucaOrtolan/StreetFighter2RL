"""
utils.py — Shared utilities for environment creation and parallel training.

Contents
--------
make_env             – Factory function that builds a single monitored SF2 env.
linear_schedule      – Returns a callable LR/clip-range schedule for PPO.
_worker              – Target function for each subprocess in SubprocVecEnvCL.
SubprocVecEnvCL      – Subclassed SubprocVecEnv that supports hot-swapping
                        game states at runtime for curriculum learning.
"""

import stable_retro as retro
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, CloudpickleWrapper
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from typing import Any, Optional
from wrapper import SFWrapper
import multiprocessing as mp
import os


# Paths are relative to the project root (one level above this file's directory).
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/states")

def make_env(game, state, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, num_stack=12, num_step_frames=8):
    """
    Return a zero-argument factory function that builds a fully wrapped SF2 env.

    The returned callable (``_init``) is what SubprocVecEnv / SubprocVecEnvCL
    calls in each worker subprocess to instantiate its own copy of the environment.
    Using a closure avoids pickling the live environment object.

    The environment stack is:
        retro.make()            ← raw emulator, provides game RAM + frame buffer
          └─ SFWrapper          ← frame stacking, action shaping, reward shaping
               └─ Monitor       ← logs episode stats (return, length, custom keys)

    Parameters
    ----------
    game : str
        stable-retro game ID, e.g. "StreetFighterIISpecialChampionEdition-Genesis-v0".
    state : str
        Path to the .state save-file to load at reset (sets matchup + difficulty).
    side : str
        "left" or "right" — which player the RL agent controls.
    reset_type : str
        "round", "match", or "never" — episode reset granularity.
    rendering : bool
        Open a display window if True (slow; use only for debugging / demos).
    init_level : int
        Arcade ladder level to start from (1 = easiest).
    state_dir : str or None
        Directory used by SFWrapper.save_state_to_file(); not used during normal training.
    verbose : bool
        If True, SFWrapper prints round/match outcomes to stdout.
    enable_combo : bool
        Expose special-move combo bits in the action space.
    null_combo : bool
        Allocate combo bits but always send a no-op (for ablation studies).
    transform_action : bool
        Use a discrete (MultiDiscrete) action space instead of MultiBinary.
    num_stack : int
        Number of frames to keep in the frame-stack buffer.
    num_step_frames : int
        Number of emulator frames per agent step (must equal len(SF_COMBOS[i])).

    Returns
    -------
    Callable[[], gymnasium.Env]
        A zero-argument factory that constructs and returns the wrapped env.
    """
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            render_mode = "human" if rendering else False,
            obs_type=retro.Observations.IMAGE,
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action, num_stack=num_stack, num_step_frames=num_step_frames)

        # Monitor wraps the env to record episode statistics to disk and expose
        # them via the `info["episode"]` key that our callbacks read.
        env = Monitor(env, LOG_DIR, info_keywords=("matches_won", "enemy_matches_won", "health", "enemy_health"))
        return env
    return _init

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):
    """
    Build a linear annealing schedule for PPO's `learning_rate` or `clip_range`.

    SB3 accepts a callable ``f(progress) → float`` where ``progress`` is the
    *remaining* training fraction (1.0 at the start, 0.0 at the end).  This
    function returns such a callable that linearly interpolates from
    ``initial_value`` (at progress=1) down to ``final_value`` (at progress=0).

    Example
    -------
    >>> schedule = linear_schedule(5e-5, 2.5e-6)
    >>> schedule(1.0)   # start of training
    5e-05
    >>> schedule(0.5)   # halfway through training
    2.625e-05
    >>> schedule(0.0)   # end of training
    2.5e-06

    Parameters
    ----------
    initial_value : float or str
        Value at the beginning of training (progress=1.0).
    final_value : float or str
        Value at the end of training (progress=0.0).  Defaults to 0.0.

    Returns
    -------
    Callable[[float], float]
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value   = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        # progress: 1.0 → 0.0 as training advances.
        return final_value + progress * (initial_value - final_value)

    return scheduler

# _worker function for custom SubprocVecEnv
def _worker(  # noqa: C901
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper
    ) -> None:
    """
    Worker function that runs inside each subprocess of SubprocVecEnvCL.

    Communicates with the main process via a multiprocessing Pipe, receiving
    commands and sending results back.  Each command is a (cmd, data) tuple.

    Supported commands (in addition to the standard SB3 set):
    ----------------------------------------------------------
    "step"         – env.step(data) → send (obs, rew, done, info, reset_info)
    "reset"        – env.reset(seed, options) → send (obs, reset_info)
    "render"       – env.render() → send rendered frame
    "close"        – close env and pipe, then exit loop
    "get_spaces"   – send (observation_space, action_space)
    "env_method"   – call a named method via get_wrapper_attr; send return value
    "get_attr"     – get a named attribute; send value
    "has_attr"     – check if attribute exists; send bool
    "set_attr"     – setattr(env, name, value)
    "is_wrapped"   – check if env is wrapped with a given wrapper class; send bool
    "update_state" – load a new save-state for curriculum learning (custom command)

    Parameters
    ----------
    remote : mp.connection.Connection
        The subprocess end of the pipe (used for send/recv in this worker).
    parent_remote : mp.connection.Connection
        The main-process end of the pipe (closed immediately in the worker
        to avoid resource leaks across forks).
    env_fn_wrapper : CloudpickleWrapper
        Cloudpickle-serialised factory callable; call .var() to instantiate the env.
    """

    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()  # Worker doesn't need the parent's pipe end.
    env = _patch_env(env_fn_wrapper.var())  # Instantiate + patch for Gym compat.
    reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)

                # Flatten terminated + truncated into a single 'done' flag (SB3 VecEnv API).
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                # Exit the worker loop and let the process terminate.
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            # update state
            elif cmd == "update_state":
                # Custom command for curriculum learning: load a new save-state
                # without restarting the worker process.
                # env.env.env.env navigates through:
                #   Monitor → SFWrapper → FrameStackObservation → retro.RetroEnv
                # and calls load_state() on the underlying retro environment.
                env.env.env.env.load_state(data)     
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            # Pipe was closed from the other end; gracefully exit.
            break
        except KeyboardInterrupt:
            # Allow Ctrl-C to propagate cleanly without error messages.
            break


# SubprocVecEnv for Curriculum Learning
class SubprocVecEnvCL(SubprocVecEnv):
    """
    Vectorised environment with curriculum learning state updates.

    Extends SB3's SubprocVecEnv to support the custom "update_state" worker
    command, allowing the main process to hot-swap the game save-state in one
    or more worker processes without restarting them.

    The custom `_worker` function (above) must be used instead of SB3's default
    worker, which is achieved by overriding `__init__` to spawn processes with
    our custom target.

    Parameters
    ----------
    env_fns : list[Callable]
        List of zero-argument factory functions, one per parallel environment.
    start_method : str or None
        Multiprocessing start method ("forkserver", "spawn", or "fork").
        Defaults to "forkserver" if available, otherwise "spawn".
        "fork" is intentionally avoided as it is not thread-safe.
    """
    def __init__(self, env_fns, start_method = None):
        self.env_list = env_fns
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        # Create one bidirectional Pipe per environment worker.
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            # preventing zombie processes if training crashes mid-run.
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()# Main process doesn't need the worker-side pipe end.

        # Initialise VecEnv base class with spaces queried from the first worker.
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)


    def update_env(self, states: list[Any], indices: VecEnvIndices = None) -> None:
        """
        Send a new save-state to one or more worker environments.

        Can be called from a CurriculumLearningCallback to advance the
        training difficulty by replacing the current game state with a harder one.

        Parameters
        ----------
        states : list
            List of state file paths (or state objects), one per targeted environment.
            Must be the same length as `indices`.
        indices : int, list[int], or None
            Indices of workers to update.  If None, update all workers, pairing
            each worker with states[0..n_envs-1].
        """
        if indices is None:
            indices = list(range(len(self.remotes)))
        else:
            indices = self._get_indices(indices)

        assert len(states) == len(indices), "Length of states must match length of indices"

        target_remotes = [self.remotes[i] for i in indices]
        for remote, state in zip(target_remotes, states):
            # The worker's _worker() loop handles "update_state" by calling
            # env.env.env.env.load_state(state) on the underlying retro env.
            remote.send(("update_state", state))

