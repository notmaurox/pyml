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

# Class that manages the Value Iteration dynamic programming algorithm for filling out the V values of a RaceCar and
# setting acceleration policy.
class RaceTrackValueIteration(object):

    # Algorithm needs a RaceTrack, A RaceCar, the learning_rate, the maximum allowed iterations through the state space
    # an a boolean to set the crash policy. If True, a crash sets the RaceCar to the start, if False, a crash places
    # a racecar at it's last non-wall position with zero velocity.
    def __init__(self, race_track: RaceTrack, race_car: RaceCar, learning_rate: float, max_iterations: int,
            crash_means_restart: bool
        ):
        self.rt = race_track
        self.rc = race_car
        self.lr = learning_rate
        self.mi = max_iterations
        self.crash_means_restart = crash_means_restart

    # Used to calculate the reward of taking an action at a state. For this model, all actions taken at non finish line
    # spaces have a cost of -1 and all finish line spaces have a cost of 0 making them an absorbing state.
    def reward(self, x, y):
        if self.rt.track_matrix[y][x] == "F":
            return 0
        else:
            return -1

    # Takes a current state in the form of it's x and y corrdinate, current x velocity, and curernt y velocity.
    # It then applies an accelration action to the state and returns the new state.
    def calc_action_outcome(self, state_x, state_y, x_velocity, y_velocity, action_x_acc_delta, action_y_acc_delta):
        # The racecar only applies the acceleration change to it's current position and velocity and returns it's end
        # state. It knows nothing about the track structure.
        new_state = self.rc.apply_acc_change_get_state(
            state_x, state_y, x_velocity, y_velocity, action_x_acc_delta, action_y_acc_delta
        )
        # The rules of this model restrict that velocity in x and y dimension cannot be outside the range of -5 to +5.
        if new_state.x_velocity > 5 or new_state.x_velocity < -5 or new_state.y_velocity > 5 or new_state.y_velocity < -5:
            return None
        # Check path from old pos to new pos by asking the track if the car is allowed to travel from previous state
        # to new state
        final_x_pos, final_y_pos = self.rt.move_along_line(state_x, state_y, new_state.x_pos, new_state.y_pos)
        # If the track says the car ends at a position that is different from the new state, it has hit a wall.
        if new_state.x_pos != final_x_pos or new_state.y_pos != final_y_pos:
            # You've hit a wall...
            if self.crash_means_restart: # Crash means restart at start position
                return State(self.rc.start_x, self.rc.start_y, 0, 0)
            else: # Crash means go to last legal position
                return State(final_x_pos, final_y_pos, 0, 0)
        # There was no collision and new state is legal.
        return new_state

    # Calculate Qt(S, A)
    def calc_action_q(self, prev_t_v, state_x, state_y, x_velocity, y_velocity, action: Action):
        # Generate final state of sucessful action...
        succesful_action_result = self.calc_action_outcome(
            state_x, state_y, x_velocity, y_velocity, action.x_acc_delta, action.y_acc_delta
        )
        if succesful_action_result is None: # Action has no result as velocity exceeds limit
            return float('-inf')
        # Calculate sucessful action score contribtion which occurs 80% of the time
        sucessful_action_value = (0.80) * prev_t_v[succesful_action_result]
        # Unsucessful action will not apply any acceleration change
        unsuccesful_action_result = self.calc_action_outcome(
            state_x, state_y, x_velocity, y_velocity, 0, 0
        )
        # Calculate unsuccesful action score contribution which occurs 20% of the time..
        unsucessful_action_value = (0.20) * prev_t_v[unsuccesful_action_result]
        # Get desitnationr reward
        destination_reward = self.reward(succesful_action_result.x_pos, succesful_action_result.y_pos)
        # Return Qt(S, A)
        return destination_reward + (self.lr * (sucessful_action_value + unsucessful_action_value))

    # Used to perform the value iteration algorithm.
    def learn_policy(self):
        check_state = State(-1, 6, 0, 0) # Used for debugging...
        for time_state in range(1, self.mi+1): # Do max_iterations number of passes over the state space
            LOG.info(f"Computing state space iteration: {time_state}")
            old_values = self.rc.v.copy() # Copy the current state of V for current set of calculations..
            for state in self.rt.legal_states(): # Iterate through all states the race car can take on the race track
                best_action, best_action_q = None, float('-inf') # Used to store the best action seen so far
                if state == check_state:
                    LOG.debug(f"Populating @ time {time_state}: {state} {self.rt.track_matrix[state.y_pos][state.x_pos]}")
                # Iterate through all actions
                for action in self.rc.actions():
                    # Calculate Qt(S, A)
                    action_q_score = self.calc_action_q(
                        old_values, state.x_pos, state.y_pos, state.x_velocity, state.y_velocity, action
                    )
                    if state == check_state:
                        LOG.debug(f"{action} has Qt(S, A) score {action_q_score}")
                    # Save the best one
                    if action_q_score > best_action_q:
                        best_action = action
                        best_action_q = action_q_score
                # Save the best action and it's score as the acceleration policy and V value at current State
                self.rc.acceleration_policy[state] = best_action
                self.rc.v[state] = best_action_q
                if state == check_state:
                    LOG.debug(f"{state} V value updated to {best_action_q} from action {action}")
        # Once the policy is learned - print the car's path through the track.
        self.animate_car_policy()

    # Used to animate the car moving through the track with the current policy..
    def animate_car_policy(self):
        max_allowed_steps = 400
        time_state = 1
        x_velocity, y_velocity = 0, 0
        x_pos, y_pos = self.rc.start_x, self.rc.start_y
        state = State(x_pos, y_pos, x_velocity, y_velocity) # Get start state
        track = self.rt.track_matrix.copy()
        has_finished = False
        while True:
            # Only allow 400 steps in an attempt to reach the finish line
            if time_state == max_allowed_steps or state is None:
                print(f"Unable to reach finish line after {time_state} iterations")
                return
            os.system('cls' if os.name == 'nt' else 'clear') # Clear terminal for printing of track
            if track[state.y_pos][state.x_pos] == "F": # You've reached the finish line
                print(f"Reached finish line after {time_state} steps")
                has_finished = True
            track[state.y_pos][state.x_pos] = str(time_state)[-1] # Update the track that a spot has been reached
            for row in track: # Print track..
                print(row)
            if has_finished: # If finish line has been reached, exit
                return
            policy_state_action = self.rc.acceleration_policy[state] # Get best action from policy
            # Allow for 20% chance that action fails.
            non_deterministic_factor = random() 
            if non_deterministic_factor <= 0.20:
                print(f"At time {time_state} ", "from", state, "action policy failed [non-deterministic result]")
                state = self.calc_action_outcome(
                    state.x_pos, state.y_pos, state.x_velocity, state.y_velocity,
                    0, 0
                )
            else: # Take action stored from policy...
                print(f"At time {time_state} ", "from", state, "took", self.rc.acceleration_policy[state])
                state = self.calc_action_outcome(
                    state.x_pos, state.y_pos, state.x_velocity, state.y_velocity,
                    policy_state_action.x_acc_delta, policy_state_action.y_acc_delta
                )
            time_state += 1

# Class that extends ValueIteration implementation to utilize it's RaceCar/RaceTrack management but overwrites it's
# learn_policy method to implement Q learning algorithm.
class QLearning(RaceTrackValueIteration):

    # Similar to value iteration with the addition of discount rate
    def __init__(self, race_track: RaceTrack, race_car: RaceCar, learning_rate: float, max_episodes: int,
            crash_means_restart: bool, discount_rate: float
        ):
        self.rt = race_track
        self.rc = race_car
        self.lr = learning_rate
        self.dr = discount_rate 
        self.curr_episode = 0
        self.me = max_episodes
        self.epsilon = 0.95 # Baseline epsilon for epsilon greedy selection of next action
        self.crash_means_restart = crash_means_restart

    # This method take the Q(S, A) table/dictionary and populates the policy with the A that maximizes Q(S, A) for 
    # each S.
    def populate_car_policy_from_q(self, q):
        for state in q.keys():
            self.rc.acceleration_policy[state] = max(q[state], key=q[state].get)

    # Calculates the update of Q(S, A) using score of taking action at previous state and taking the action
    # with highest score at next state.
    def calculate_q_update(self, q, prev_state, new_state, action):
        reward = -1 # This calculation only happens when not at finish line state so reward is a cost of -1
        action_score = q[prev_state][action]
        # Action at new state is chosen using the one with highest Q score at next state.
        return ( action_score
                 + (self.lr*(reward + self.dr*max(q[new_state].values()) - action_score))
        )

    # Epsilon greedy selection of action at current state..
    def pick_next_action(self, q, state):
        # Start with high value epsilon and gradually decrease it as more episodes occur...
        working_epsilon = ((self.me - self.curr_episode) / self.me) * 0.95
        rand_float = random() # Returns random number between 0 and 1...
        # with epislon probability, choose a random action...
        if rand_float <= working_epsilon:
            return np.random.choice(list(q[state]))
        else: # with 1-epsilon probability, choose the best action...
            return max(q[state], key=q[state].get)

    # Initialize q table/dictionary with random values for each action at each legal state.
    def initialize_q(self):
        q = {}
        for state in self.rt.legal_states():
            q[state] = {}
            for action in self.rc.actions():
                q[state][action] = random()
        return q

    # Implementation of Q learning algorithm.
    def learn_policy(self):
        # Initialize all Q(s, a) arbitrarily.
        q = self.initialize_q()
        for episode_iteration in range(self.me): # Do maximum_iterations number of episodes.
            self.curr_episode = episode_iteration
            os.system('cls' if os.name == 'nt' else 'clear')
            LOG.info(f"Performing episode iteration: {episode_iteration} / {self.me}")
            # Reset end state Q scores...
            for end_state in self.rt.end_states():
                q[end_state] = {}
                for action in self.rc.actions():
                    q[end_state][action] = 0.0
            # Start from RaceCars start position.
            state = State(self.rc.start_x, self.rc.start_y, 0, 0)
            while True:
                # If the race car has reached the finish - stop current episode.
                if self.rt.track_matrix[state.y_pos][state.x_pos] == "F":
                    break
                # Greedy policy to pick best action at state...
                next_action = self.pick_next_action(q, state)
                # Calculate the outcome of taking next_action at current state. calc_action_outcome will return None
                # If the action taken at current state breaks velocity outside of -5 to +5 range rule.
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
        self.populate_car_policy_from_q(q) # Populate car policy from Q
        self.animate_car_policy() # Animate car policy

# Class that extends QLearning implementation to utilize it's RaceCar/RaceTrack management in addition to learning
# protocol
class SARSA(QLearning):

    # The only difference between Q learning and SARSA is how the Q score update of Q(S, A) is calculated. 
    # Here it uses the same protocol to chose the next action in the next state as it did to chose the current action
    # at the previous state. This is different from Q learning where the next action is chosen as the one with max Q
    # score at next state.
    def calculate_q_update(self, q, prev_state, new_state, action):
        reward = -1 # This calculation only happens when not at finish line state so reward is a cost of -1
        action_score = q[prev_state][action]
        next_state_action = self.pick_next_action(q, new_state) # Pick an action at the next state using epsilon greedy
        return ( action_score
                 + (self.lr*(reward + self.dr*q[new_state][next_state_action] - action_score))
        )
