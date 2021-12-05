import os
import pathlib
from collections import defaultdict

PATH_TO_DATA_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "../", "../", "data/", "racetracks/")

class RaceTrackLoader(object):

    @staticmethod
    def LoadLTrack():
        path_to_L_track = os.path.join(PATH_TO_DATA_DIR, "L-track.txt")
        return RaceTrack(path_to_L_track)

    @staticmethod
    def LoadRTrack():
        path_to_R_track = os.path.join(PATH_TO_DATA_DIR, "R-track.txt")
        return RaceTrack(path_to_R_track)

    @staticmethod
    def LoadOTrack():
        path_to_O_track = os.path.join(PATH_TO_DATA_DIR, "O-track.txt")
        return RaceTrack(path_to_O_track)

class RaceTrack(object):

    def __init__(self, path_to_racetrack_file: str):
        self.track_path = path_to_racetrack_file
        self.track_matrix = []
        with open(self.track_path, 'r') as track_file:
            file_lines = track_file.readlines()
            for line_index in range(len(file_lines)):
                line = file_lines[line_index].strip()
                if line_index == 0:
                    line_split = line.split(",")
                    self.height, self.length = int(line_split[0]), int(line_split[1])
                else:
                    self.track_matrix.append(
                        list(line)
                    )

    # Return last non wall position in straight line from (x1, y1) to (x2, y2)
    def move_along_line(self, x1, y1, x2, y2):
        points_in_path = self.points_in_path(x1, y1, x2, y2)
        for point in points_in_path:
            x, y =  point[0], point[1]
            if self.track_matrix[y][x] == "#":
                # Hit a wall.
                return prev_x, prev_y
            if self.track_matrix[y][x] == "F":
                # Hit the finish
                return x, y
            prev_x, prev_y = x, y
        return x2, y2

    def points_in_path(self, x1, y1, x2, y2):
        # Moving only in y dimension
        if x1 == x2:
            if y1 < y2:
                y_range = list(range(y1, y2+1))
            else:
                y_range = list(range(y2, y1+1))
                y_range.reverse()
            return [(x1, y_pos) for y_pos in y_range]
        # Moving only in x dimension
        if y2 == y1:
            if x1 < x2:
                x_range = list(range(x1, x2+1))
            else:
                x_range = list(range(x2, x1+1))
                x_range.reverse()
            return [(x_pos, y1) for x_pos in x_range]
        # Moving in both x and y dimensions
        x_delta, y_delta = x2 - x1, y2 - y1
        if x_delta > 0:
            x_sign = 1
        else:
            x_sign = -1

        if y_delta > 0:
            y_sign = 1
        else:
            y_sign = -1

        x_delta, y_delta = abs(x_delta), abs(y_delta)

        if x_delta > y_delta:
            x_dx, x_dy = x_sign, 0
            y_dx, y_dy = 0, y_sign
        else:
            x_delta, y_delta = y_delta, x_delta
            x_dx, x_dy = 0, y_sign
            y_dx, y_dy = x_sign, 0

        delta = 2*y_delta - x_delta
        y = 0
        to_return = []
        for x in range(x_delta + 1):
            x_coordinate = x1 + x*x_dx + y*y_dx
            y_coordinate = y1 + x*x_dy + y*y_dy
            to_return.append((x_coordinate, y_coordinate))
            if delta >= 0:
                y += 1
                delta -= 2*x_delta
            delta += 2*y_delta
        return to_return

    def legal_states(self):
        for y in range(len(self.track_matrix)):
            for x in range(len(self.track_matrix[y])):
                if self.track_matrix[y][x] == "." or self.track_matrix[y][x] == "S":
                    for x_velocity in range(-5, 6):
                        for y_velocity in range(-5, 6):
                            yield State(x, y, x_velocity, y_velocity)

    def end_states(self):
        for y in range(len(self.track_matrix)):
            for x in range(len(self.track_matrix[y])):
                if self.track_matrix[y][x] == "F":
                    for x_velocity in range(-5, 6):
                        for y_velocity in range(-5, 6):
                            yield State(x, y, x_velocity, y_velocity)

    def start_positions(self):
        start_positions = []
        for y in range(len(self.track_matrix)):
            for x in range(len(self.track_matrix[y])):
                if self.track_matrix[y][x] == "S":
                    start_positions.append(
                        (x, y)
                    )
        return start_positions

    def print_track(self):
        for row in self.track_matrix:
            print(row)

class State(object):

    def __init__(self, x_pos, y_pos, x_velocity, y_velocity):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity

    def __str__(self):
        return f"State ({self.x_pos}, {self.y_pos}) with vx: {self.x_velocity} vy: {self.y_velocity}"

    def __eq__(self, other):
        return all([
            self.x_pos == other.x_pos, self.y_pos == other.y_pos,
            self.x_velocity == other.x_velocity, self.y_velocity == other.y_velocity
        ])

    def __hash__(self):
        return hash((self.x_pos, self.y_pos, self.x_velocity, self.y_velocity))

class Action(object):

    def __init__(self, x_acc_delta, y_acc_delta):
        self.x_acc_delta = x_acc_delta
        self.y_acc_delta = y_acc_delta

    def __str__(self):
        return f"Action dxv: {self.x_acc_delta} dxy {self.y_acc_delta}"

class RaceCar(object):

    def __init__(self, track, start_x, start_y):
        # self.policy = [[0 for _ in range(track.length)] for _ in range(track.height)]
        self.start_x, self.start_y = start_x, start_y
        self.v = {state: 0 for state in track.legal_states()}
        for end_state in track.end_states():
            self.v[end_state] = 0
        self.acceleration_policy = {}

    def actions(self):
        for x_acc_delta in [-1, 0, 1]:
            for y_acc_delta in [-1, 0, 1]:
                yield Action(x_acc_delta, y_acc_delta)
    
    def apply_acc_change_get_state(self, x_start, y_start, x_velocity, y_velocity, x_acc_delta, y_acc_delta):
        new_x_velocity = x_velocity + x_acc_delta
        new_y_velocity = y_velocity + y_acc_delta
        new_x = x_start + new_x_velocity
        new_y = y_start + new_y_velocity
        return State(new_x, new_y, new_x_velocity, new_y_velocity)

    def print_v(self, track):

        for row in self.v:
            print(row)

if __name__ == "__main__":
    rt = RaceTrack("../../data/racetracks/L-track.txt")
    rt.print_track()

    rc = RaceCar(rt)