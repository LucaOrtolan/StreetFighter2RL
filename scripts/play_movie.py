import stable_retro
import os
import time  # Add at top

from const import sf_game
movie = stable_retro.Movie(os.path.join(os.path.dirname(os.path.dirname(__file__)), "replays/StreetFighterIISpecialChampionEdition-Genesis-v0-ryu_vs_ken_8-000000.bk2"))
movie.step()

env = stable_retro.make(
    game=movie.get_game(),
    state=None,
    render_mode='human',
    # bk2s can contain any button presses, so allow everything
    use_restricted_actions=stable_retro.Actions.ALL,
    players=movie.players,
)
env.initial_state = movie.get_state()
env.reset()

# In the while loop:
while movie.step():
    keys = []
    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))
    env.step(keys)
    time.sleep(1/60.0)  # ~16.7ms for 60 FPS; adjust as needed (e.g., 1/30 for half speed)
