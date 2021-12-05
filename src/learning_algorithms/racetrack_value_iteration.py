import logging
import sys
import os
import random
import numpy as np
from random import random

from data_utils.race_track import RaceTrack, RaceCar, Action, State

# Logging stuff
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG.addHandler(handler)

class ActionResult(object):

    def __init__(self, x_pos, y_pos, x_velocity, y_velocity):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity

class RaceTrackValueIteration(object):

    def __init__(self, race_track: RaceTrack, race_car: RaceCar, learning_rate: float, max_iterations: int,
            crash_means_restart: bool
        ):
        self.rt = race_track
        self.rc = race_car
        self.lr = learning_rate
        self.mi = max_iterations
        self.crash_means_restart = crash_means_restart

    def reward(self, x, y):
        if self.rt.track_matrix[y][x] == "F":
            "finished"
            return 0
        else:
            return -1

    def calc_action_outcome(self, state_x, state_y, x_velocity, y_velocity, action_x_acc_delta, action_y_acc_delta):
        new_state = self.rc.apply_acc_change_get_state(
            state_x, state_y, x_velocity, y_velocity, action_x_acc_delta, action_y_acc_delta
        )
        if new_state.x_velocity > 5 or new_state.x_velocity < -5 or new_state.y_velocity > 5 or new_state.y_velocity < -5:
            return None
        # Check path from old pos to new pos...
        final_x_pos, final_y_pos = self.rt.move_along_line(state_x, state_y, new_state.x_pos, new_state.y_pos)
        if new_state.x_pos != final_x_pos or new_state.y_pos != final_y_pos:
            # You've hit a wall...
            if self.crash_means_restart:
                return State(self.rc.start_x, self.rc.start_y, 0, 0)
            else:
                return State(final_x_pos, final_y_pos, 0, 0)
        return new_state

    def calc_action_q(self, prev_t_v, state_x, state_y, x_velocity, y_velocity, action: Action):
        succesful_action_result = self.calc_action_outcome(
            state_x, state_y, x_velocity, y_velocity, action.x_acc_delta, action.y_acc_delta
        )
        if succesful_action_result is None: # Action has no result as velocity exceeds limit
            return float('-inf')
        sucessful_action_value = (0.80) * prev_t_v[succesful_action_result]
        unsuccesful_action_result = self.calc_action_outcome(
            state_x, state_y, x_velocity, y_velocity, 0, 0
        )
        unsucessful_action_value = (0.20) * prev_t_v[unsuccesful_action_result]
        destination_reward = self.reward(succesful_action_result.x_pos, succesful_action_result.y_pos)
        return destination_reward + (self.lr * (sucessful_action_value + unsucessful_action_value))

    def learn_policy(self):
        check_state = State(-1, 6, 0, 0)
        for time_state in range(1, self.mi+1):
            LOG.info(f"Computing state space iteration: {time_state}")
            old_values = self.rc.v.copy()
            for state in self.rt.legal_states():
                best_action, best_action_q = None, float('-inf')
                if state == check_state:
                    print(f"Populating @ time {time_state}:", state, self.rt.track_matrix[state.y_pos][state.x_pos])
                for action in self.rc.actions():
                    action_q_score = self.calc_action_q(
                        old_values, state.x_pos, state.y_pos, state.x_velocity, state.y_velocity, action
                    )
                    if state == check_state:
                        print(action, "Action score ", action_q_score)
                    if action_q_score > best_action_q:
                        best_action = action
                        best_action_q = action_q_score
                self.rc.acceleration_policy[state] = best_action
                self.rc.v[state] = best_action_q
            # exit()
        self.animate_car_policy()

    def animate_car_policy(self):
        max_allowed_steps = 400
        time_state = 1
        x_velocity, y_velocity = 0, 0
        x_pos, y_pos = self.rc.start_x, self.rc.start_y
        state = State(x_pos, y_pos, x_velocity, y_velocity)
        track = self.rt.track_matrix.copy()
        has_finished = False
        while True:
            if time_state == max_allowed_steps:
                print(f"Unable to reach finish line after {max_allowed_steps} iterations")
            os.system('cls' if os.name == 'nt' else 'clear')
            if track[state.y_pos][state.x_pos] == "F":
                print(f"Reached finish line after {time_state} steps")
                has_finished = True
            else:
                print(f"At time {time_state} ", "from", state, "took", self.rc.acceleration_policy[state])
            track[state.y_pos][state.x_pos] = str(time_state)[-1]
            for row in track:
                print(row)
            if has_finished:
                return
            policy_state_action = self.rc.acceleration_policy[state]
            state = self.calc_action_outcome(
                state.x_pos, state.y_pos, state.x_velocity, state.y_velocity,
                policy_state_action.x_acc_delta, policy_state_action.y_acc_delta
            )
            time_state += 1

class QLearning(RaceTrackValueIteration):

    def __init__(self, race_track: RaceTrack, race_car: RaceCar, learning_rate: float, max_episodes: int,
            crash_means_restart: bool, discount_rate: float
        ):
        self.rt = race_track
        self.rc = race_car
        self.lr = learning_rate
        self.dr = discount_rate 
        self.curr_episode = 0
        self.me = max_episodes
        self.epsilon = 0.95
        self.crash_means_restart = crash_means_restart

    def populate_car_policy_from_q(self, q):
        for state in q.keys():
            self.rc.acceleration_policy[state] = max(q[state], key=q[state].get)

    def calculate_q_update(self, q, prev_state, new_state, action):
        reward = -1
        action_score = q[prev_state][action]
        return ( action_score
                 + (self.lr*(reward + self.dr*max(q[new_state].values()) - action_score))
        )

    def pick_next_action(self, q, state):
        # start with high value epsilon and gradually decrease it...
        working_epsilon = ((self.me - self.curr_episode) / self.me) * 0.95
        rand_float = random()
        # with epislon probability, choose a random action...
        if rand_float < working_epsilon:
            return np.random.choice(list(q[state]))
        else: # with 1-epsilon probability, choose the best action...
            return max(q[state], key=q[state].get)

    def learn_policy(self):
        # Initialize all Q(s, a) arbitrarily.
        q = {}
        for state in self.rt.legal_states():
            q[state] = {}
            for action in self.rc.actions():
                q[state][action] = random()
        for episode_iteration in range(self.me):
            self.curr_episode = episode_iteration
            os.system('cls' if os.name == 'nt' else 'clear')
            LOG.info(f"Performing episode iteration: {episode_iteration} / {self.me}")
            # Reset end state Q scores...
            for end_state in self.rt.end_states():
                q[end_state] = {}
                for action in self.rc.actions():
                    q[end_state][action] = 0.0
            # Start from someplace random
            # state = np.random.choice(list(self.rt.legal_states()))
            state = State(self.rc.start_x, self.rc.start_y, 0, 0)
            while True:
                if self.rt.track_matrix[state.y_pos][state.x_pos] == "F":
                    break
                # Greedy policy to pick best action at state...
                next_action = self.pick_next_action(q, state)
                # Take best action at current state. If calc_action_outcome returns None, that action is not legal at
                # current state
                new_state = self.calc_action_outcome(
                    state.x_pos, state.y_pos, state.x_velocity, state.y_velocity,
                    next_action.x_acc_delta, next_action.y_acc_delta
                )
                # Pick actions untill one with legal consequence is chosen.
                while new_state is None:
                    q[state].pop(next_action) # next_action is illegal at current state so it should be removed
                    next_action = self.pick_next_action(q, state)
                    new_state = self.calc_action_outcome(
                        state.x_pos, state.y_pos, state.x_velocity, state.y_velocity,
                        next_action.x_acc_delta, next_action.y_acc_delta
                    )
                # Update Q with results of action
                q[state][next_action] = self.calculate_q_update(q, state, new_state, next_action)
                # Update state
                state = new_state
        self.populate_car_policy_from_q(q)
        self.animate_car_policy()

class SARSA(QLearning):

    def calculate_q_update(self, q, prev_state, new_state, action):
        reward = -1
        action_score = q[prev_state][action]
        next_state_action = self.pick_next_action(q, new_state)
        return ( action_score
                 + (self.lr*(reward + self.dr*q[new_state][next_state_action] - action_score))
        )
