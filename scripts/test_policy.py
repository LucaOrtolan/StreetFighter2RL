from stable_baselines3 import PPO
from wrapper import SFWrapper
import stable_retro as retro
import os
from const import *
from collections import deque

def load_model(model_name):
    saved_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")

    model = PPO.load(os.path.join(saved_models_path, model_name))

    return model

def make_env(state,
            game=sf_game,
            side="left", 
            reset_type="round", 
            init_level=1,
            rendering=False,
            num_stack=12, 
            num_step_frames=8,
            state_dir=None,
            verbose=False,
            enable_combo=True,
            null_combo=False,
            transform_action=False,
            record=False):
        
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        record=record,
        render_mode="human" if rendering else False
    )
    
    env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action, num_stack=num_stack, num_step_frames=num_step_frames)
    return env

def play(env, model, episodes):
    # Reset game to starting state
    obs = env.reset()
    # Set flag to flase
    frames = deque(maxlen=env.num_stack)
    wins = 0
    for ep in range(episodes): 
        print(f"episode = {ep}")
        game_on = True
        while game_on:
            if len(frames) < env.num_stack:
                action = np.zeros(env.action_dim, dtype=np.int8)
            else:
                action = model.predict(obs)[0]

            obs, reward, done, truncated, info = env.step(action)
            frames.append(obs)
            if info["matches_won"]==2 or info["enemy_matches_won"]==2:
                if info["matches_won"] == 2:
                    wins+=1
                print(f"Player: {info['matches_won']} - CPU: {info['enemy_matches_won']}")
                env.reset()
                game_on = False


    env.close()    
    print("Simulation completed. Final winrate = %.2f"%(wins/episodes))
    print(f"Wins = {wins} out of {episodes} matches")


def main():
    model_name = "best_model_400000"
    state = "/home/master26/Documents/StreetFighter2RL/FightLadder/data/sf/curriculum/Level2.20.state"
    # record = os.path.join(os.path.dirname(os.path.dirname(__file__)), "replays")
    record = False
    rendering=False
    episodes = 10

    env = make_env(state=state, rendering=rendering, record=record)
    model = load_model(model_name)
    play(env, model, episodes)


if __name__=="__main__":
    main()