import os
import pathlib
from collections import defaultdict

PATH_TO_DATA_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "data/", "racetracks/")

# Object that functions to store the combination of a state on the racetrack occupied by a racecar that contains an
# x coordinate, y coordinate, current x velocity, and current y velocity.
class State(object):

    # Each state has an x position, y position, current x velocity, and current y velocity.
    def __init__(self, x_pos, y_pos, x_velocity, y_velocity):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity

    # Used for printing the state
    def __str__(self):
        return f"State ({self.x_pos}, {self.y_pos}) with vx: {self.x_velocity} vy: {self.y_velocity}"

    # Used for comparing if two states contain equivalent information, useful for debugging
    def __eq__(self, other):
        return all([
            self.x_pos == other.x_pos, self.y_pos == other.y_pos,
            self.x_velocity == other.x_velocity, self.y_velocity == other.y_velocity
        ])

    # Used to hash state objects into dictionary keys
    def __hash__(self):
        return hash((self.x_pos, self.y_pos, self.x_velocity, self.y_velocity))

# An object used to represent an action that can be taken by a racecar which comes in the form of a change in x
# acceleration and a change in y acceleration
class Action(object):

    def __init__(self, x_acc_delta, y_acc_delta):
        self.x_acc_delta = x_acc_delta
        self.y_acc_delta = y_acc_delta

    # Used for printing an action...
    def __str__(self):
        return f"Action dxv: {self.x_acc_delta} dxy {self.y_acc_delta}"

# RaceTrack factory class that returns tracks of L/R/O type.
class RaceTrackLoader(object):

    # Returns a RaceTrack object for the L track
    @staticmethod
    def LoadLTrack():
        path_to_L_track = os.path.join(PATH_TO_DATA_DIR, "L-track.txt")
        return RaceTrack(path_to_L_track)

    # Returns a RaceTrack object for the R track
    @staticmethod
    def LoadRTrack():
        path_to_R_track = os.path.join(PATH_TO_DATA_DIR, "R-track.txt")
        return RaceTrack(path_to_R_track)

    # Returns a RaceTrack object for the O track
    @staticmethod
    def LoadOTrack():
        path_to_O_track = os.path.join(PATH_TO_DATA_DIR, "O-track.txt")
        return RaceTrack(path_to_O_track)

# Class that handles storing a race track as a matrix and coordinating actions taken on the racetrack
class RaceTrack(object):

    # Class constructor that takes a path to a racetrack file and loads it into a matrix
    def __init__(self, path_to_racetrack_file: str):
        self.track_path = path_to_racetrack_file
        self.track_matrix = []
        with open(self.track_path, 'r') as track_file:
            file_lines = track_file.readlines()
            # Each row in the racetrack is stored as a list within a larger list. This means indexing locations on the
            # track requires self.track_matrix[y_coordinate][x_coordinate]
            for line_index in range(len(file_lines)):
                line = file_lines[line_index].strip()
                # The first line in the file specifies the height and length of the track
                if line_index == 0:
                    line_split = line.split(",")
                    self.height, self.length = int(line_split[0]), int(line_split[1])
                # All other lines are rows of the track itself
                else:
                    self.track_matrix.append(
                        list(line)
                    )

    # Return last non wall position in straight line from (x1, y1) to (x2, y2)
    def move_along_line(self, x1, y1, x2, y2):
        # This method returns all the points along the path between (x1, y1), and (x2, y2)
        points_in_path = self.points_in_path(x1, y1, x2, y2)
        # Iterate through all the points on the track.
        for point in points_in_path:
            x, y =  point[0], point[1]
            # If you've hit a wall, return the last point before the wall
            if self.track_matrix[y][x] == "#":
                # Hit a wall.
                return prev_x, prev_y
            # If you've hit the finish line, return the point on the finish line it's hit.
            if self.track_matrix[y][x] == "F":
                # Hit the finish
                return x, y
            # Store the last seen valid x and y coodrinate
            prev_x, prev_y = x, y
        # None of the points in the path hit a wall or a finish so (x2, y2) can be sucessfully reached
        return x2, y2

    # Helper method to move_alone_line that generates all the points along a line that goes from (x1, y1) to (x2, y2)
    def points_in_path(self, x1, y1, x2, y2):
        # Moving only in y dimension
        if x1 == x2:
            if y1 < y2: # If moving from smaller y to larger y
                y_range = list(range(y1, y2+1))
            else: # If moving from larger y to smaller y
                y_range = list(range(y2, y1+1))
                y_range.reverse()
            return [(x1, y_pos) for y_pos in y_range] # Generate list of points
        # Moving only in x dimension
        if y2 == y1:
            if x1 < x2: # If moving from smaller x to larger x
                x_range = list(range(x1, x2+1))
            else: # If moving from larger x to smaller x
                x_range = list(range(x2, x1+1))
                x_range.reverse()
            return [(x_pos, y1) for x_pos in x_range] # Generate list of points
        # Moving in both x and y dimensions
        x_delta, y_delta = x2 - x1, y2 - y1
        if x_delta > 0: # Determine direction of x change
            x_sign = 1
        else:
            x_sign = -1
        if y_delta > 0: # Determine direction of y change
            y_sign = 1
        else:
            y_sign = -1
        # Determine absolute number of units to travel
        x_delta, y_delta = abs(x_delta), abs(y_delta)
        # Determine which dimension must travel the longer distance
        if x_delta > y_delta:
            x_dx, x_dy = x_sign, 0
            y_dx, y_dy = 0, y_sign
        else:
            x_delta, y_delta = y_delta, x_delta
            x_dx, x_dy = 0, y_sign
            y_dx, y_dy = x_sign, 0
        # Used to store remainder between reported and actual travel dimension distance
        delta = 2*y_delta - x_delta
        y = 0
        to_return = []
        # Iterate through all the steps in X
        for x in range(x_delta + 1):
            x_coordinate = x1 + x*x_dx + y*y_dx
            y_coordinate = y1 + x*x_dy + y*y_dy
            to_return.append((x_coordinate, y_coordinate)) # Keep list of visited points
            # If remainder between reported and actual travel dimension space is too large, take a step in the y
            if delta >= 0:
                y += 1
                delta -= 2*x_delta # Account for increase in y
            delta += 2*y_delta # Update remainder distance
        return to_return # Return visted points

    # Generator that can be used to iterate though all legal race track states. AKS coordinates of "S" and "."
    # spaces on the race track and the legal x and y velocities that a race car at those coordinates could be moving at
    def legal_states(self):
        for y in range(len(self.track_matrix)):
            for x in range(len(self.track_matrix[y])):
                if self.track_matrix[y][x] == "." or self.track_matrix[y][x] == "S":
                    for x_velocity in range(-5, 6):
                        for y_velocity in range(-5, 6):
                            yield State(x, y, x_velocity, y_velocity)

    # Generator that can be used to iterate through all end states and the various velocities a race car could be moving
    # at once reacing the goal state
    def end_states(self):
        for y in range(len(self.track_matrix)):
            for x in range(len(self.track_matrix[y])):
                if self.track_matrix[y][x] == "F":
                    for x_velocity in range(-5, 6):
                        for y_velocity in range(-5, 6):
                            yield State(x, y, x_velocity, y_velocity)

    # Returns a list of only the x and y positions that are start coordinates on the race track
    def start_positions(self):
        start_positions = []
        for y in range(len(self.track_matrix)):
            for x in range(len(self.track_matrix[y])):
                if self.track_matrix[y][x] == "S":
                    start_positions.append(
                        (x, y)
                    )
        return start_positions

    # Prints the racetrack to STD out.
    def print_track(self):
        for row in self.track_matrix:
            print(row)

# Used to track data associated with the RaceCar that is learning to navigate the track.
class RaceCar(object):

    def __init__(self, track, start_x, start_y):
        self.start_x, self.start_y = start_x, start_y # Car remembers it's start position
        self.v = {state: 0 for state in track.legal_states()} # v is a dictionary where each key is a State object
        for end_state in track.end_states():
            self.v[end_state] = 0
        # This will be a dictionary where the keys are States and the values are the Action to be taken at that state
        self.acceleration_policy = {}

    # Generator that can be used to iterate through all the acceleration actions a car can take
    def actions(self):
        for x_acc_delta in [-1, 0, 1]:
            for y_acc_delta in [-1, 0, 1]:
                yield Action(x_acc_delta, y_acc_delta)
    
    # Method for taking the current x_position, y_position, x_velocity, y_velocity, change in x acc, change in y acc
    # and return the resulting state from applying that action.
    # Note: This could be refactored to accept a State and an Action but this is more explicit about what exact pieces
    # of inormation are used in an update...
    def apply_acc_change_get_state(self, x_start, y_start, x_velocity, y_velocity, x_acc_delta, y_acc_delta):
        new_x_velocity = x_velocity + x_acc_delta
        new_y_velocity = y_velocity + y_acc_delta
        new_x = x_start + new_x_velocity
        new_y = y_start + new_y_velocity
        return State(new_x, new_y, new_x_velocity, new_y_velocity)

    # Used to Print some dimenson of V
    def print_v(self):
        for row in self.v:
            print(row)
