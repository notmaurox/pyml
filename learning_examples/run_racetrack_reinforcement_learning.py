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
    learning_type = sys.argv[1]
    track_name = sys.argv[2]
    max_iterations = int(sys.argv[3])
    crash_protocol = sys.argv[4]

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
    
    if "reset" in crash_protocol:
        reset_on_crash = True
    else:
        reset_on_crash = False

    rt.print_track()
    start_x, start_y = rt.start_positions()[0]
    rc = RaceCar(rt, start_x, start_y)

    if "value" in learning_type:
        rt_value_iteration = RaceTrackValueIteration(rt, rc, 0.80, max_iterations, reset_on_crash)
        rt_value_iteration.learn_policy()
    if "Q" in learning_type:
        q_learning = QLearning(rt, rc, learning_rate=0.25, max_episodes=max_iterations,
            crash_means_restart=reset_on_crash, discount_rate=0.90)
        q_learning.learn_policy()
    if "SARSA" in learning_type:
        sarsa = SARSA(rt, rc, learning_rate=0.25, max_episodes=max_iterations,
            crash_means_restart=reset_on_crash, discount_rate=0.90)
        sarsa.learn_policy()



