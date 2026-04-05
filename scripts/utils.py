import stable_retro as retro
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, CloudpickleWrapper
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from typing import Any, Optional
from wrapper import SFWrapper
import multiprocessing as mp
import re
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/states")

def make_env(game, state, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, num_stack=12, num_step_frames=8, rewards_scheme="default"):
    # print(
    #     f"make_env received:\n"
    #     f"  game={game}\n"
    #     f"  state={state}\n"
    #     f"  side={side}\n"
    #     f"  reset_type={reset_type}\n"
    #     f"  rendering={rendering}\n"
    #     f"  init_level={init_level}\n"
    #     f"  state_dir={state_dir}\n"
    #     f"  verbose={verbose}\n"
    #     f"  enable_combo={enable_combo}\n"
    #     f"  null_combo={null_combo}\n"
    #     f"  transform_action={transform_action}\n"
    #     f"  num_stack={num_stack}\n"
    #     f"  num_step_frames={num_step_frames}\n"
    #     f"  rewards_scheme={rewards_scheme}\n"
    # )

    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            render_mode = "human" if rendering else False,
            obs_type=retro.Observations.IMAGE,
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action, num_stack=num_stack, num_step_frames=num_step_frames, rewards_scheme=rewards_scheme)
        env = Monitor(env, LOG_DIR, info_keywords=("matches_won", "enemy_matches_won", "health", "enemy_health"))
        return env
    return _init

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def get_state(matchup, lvl, STATE_DIR=STATE_DIR):
    """Example arguments
    
        matchup = ryu_vs_ken
        lvl = 1
    """
    state_to_load = STATE_DIR + f"/{matchup}_{lvl}.state"
    return state_to_load

def extract_matchup(path):
    basename = os.path.splitext(os.path.basename(path))[0]  # e.g. "ryu_vs_ken_1"
    # matchup = re.sub(r"_\d+$", "", basename)  # -> "ryu_vs_ken"
    return basename



# _worker function for custom SubprocVecEnv
def _worker(  # noqa: C901
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper
    ) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
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
                env.env.env.env.load_state(data)     
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except KeyboardInterrupt:
            break


# SubprocVecEnv for Curriculum Learning
class SubprocVecEnvCL(SubprocVecEnv):
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

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)


    def update_env(self, states: list[Any], indices: VecEnvIndices = None) -> None:
        """
        Send a new internal state to one or more environments.

        :param states: List of states, one state per environment to update.
                    Each state should be compatible with the env's `load_state` or similar.
        :param indices: Indices of environments to update.
                        If None, update all environments with states[0:len(remotes)].
                        Must match the length of `states`.
        """
        if indices is None:
            indices = list(range(len(self.remotes)))
        else:
            indices = self._get_indices(indices)

        assert len(states) == len(indices), "Length of states must match length of indices"

        target_remotes = [self.remotes[i] for i in indices]
        for remote, state in zip(target_remotes, states):
            remote.send(("update_state", state))

