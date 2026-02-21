from wrapper import SFWrapper
from stable_baselines3.common.monitor import Monitor
import stable_retro as retro
from const import *

def make_env(game, 
            state,
            side, 
            reset_type="round", 
            init_level=1,
            rendering=False,
            num_stack=12, 
            num_step_frames=8,
            state_dir=None,
            verbose=False,
            enable_combo=True,
            null_combo=False,
            transform_action=False):
        
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
    )
    
    env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action, num_stack=num_stack, num_step_frames=num_step_frames)
    # env = Monitor(env)

    return env
    

def random_agent(env, episodes=1):
    # Reset game to starting state
    obs = env.reset()
    # Set flag to flase
    done = False

    for game in range(episodes): 
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
        print(f"Player: {info['matches_won']} - CPU: {info['enemy_matches_won']}")
    
    env.close()    

def main():
    game = 'StreetFighterIISpecialChampionEdition-Genesis-v0'
    state = "ken"
    side = "left" # side for AI to control
    reset_type = "match"
    rendering = True
    verbose = True
    enable_combo = True # enable special move action space for environment
    null_combo = False # null action space for special move
    transform_action = False
    num_stack = 12 # number of frames to stack
    num_step_frames = 8 # number of frames per step
    episodes = 1

    env = make_env(game, state=state, side=side, reset_type=reset_type, rendering=rendering, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action, num_stack=num_stack, num_step_frames=num_step_frames)


    random_agent(env, episodes)

if __name__ == "__main__":
    main()




