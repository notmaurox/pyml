import sys
import os
import pathlib
import logging
import statistics
import pandas as pd
import matplotlib.pyplot as plt

PATH_TO_SRC_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "src/")
sys.path.insert(0, PATH_TO_SRC_DIR)

from data_utils.race_track import RaceTrackLoader, RaceTrack, RaceCar
from learning_algorithms.racetrack_value_iteration import RaceTrackValueIteration, QLearning, SARSA

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

if __name__ == "__main__":
    learning_type = sys.argv[1] # First arg determines learning type, accepts value/Q/SARSA
    track_name = sys.argv[2] # Second arg determines track type. accepts L/O/R
    max_iterations = int(sys.argv[3]) # Third arg is an integer that determines the number of iterations to run
    crash_protocol = sys.argv[4] # Fourth arg determines the crash policy, if arguement is "reset", a crash resets the
    # car to it's start position...

    # Determine track type
    if "L" in track_name:
        LOG.info("Learning L shaped track")
        rt = RaceTrackLoader.LoadLTrack()
    elif "R" in track_name:
        LOG.info("Learning R shaped track")
        rt = RaceTrackLoader.LoadRTrack()
    elif "O" in track_name:
        LOG.info("Learning O shaped track")
        rt = RaceTrackLoader.LoadOTrack()
    else:
        LOG.fatal(f"{track_name} not a supported track name")
        exit()
    # Determine crash protocol
    if "reset" in crash_protocol:
        reset_on_crash = True
        LOG.info("Using crash policy: crash means RESET")
    else:
        reset_on_crash = False
        LOG.info("Using crash policy: crash means STOP")
    # Print track
    rt.print_track()
    start_x, start_y = rt.start_positions()[0] # Car will always start at the first start position of each track...
    rc = RaceCar(rt, start_x, start_y) # Initialize racecar

    # Initialize learning model
    if "value" in learning_type:
        LOG.info(f"Performing value iteration with {max_iterations} iterations through state space")
        learning_model = RaceTrackValueIteration(rt, rc, 0.80, max_iterations, reset_on_crash)
    if "Q" in learning_type:
        LOG.info(f"Performing Q learning with {max_iterations} episodes")
        learning_model = QLearning(rt, rc, learning_rate=0.25, max_episodes=max_iterations,
            crash_means_restart=reset_on_crash, discount_rate=0.90)
    if "SARSA" in learning_type:
        LOG.info(f"Performing SARSA learning with {max_iterations} episodes")
        learning_model = SARSA(rt, rc, learning_rate=0.25, max_episodes=max_iterations,
            crash_means_restart=reset_on_crash, discount_rate=0.90)
    # Learn the policy
    learning_model.learn_policy()



