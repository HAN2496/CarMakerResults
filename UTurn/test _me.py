import numpy as np


def to_relative_coordinates(x, y, yaw, lookahead_points):
    relative_coords = []

    for point in lookahead_points:
        dx = point[0] - x
        dy = point[1] - y

        rotated_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
        rotated_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)

        relative_coords.append((rotated_x, rotated_y, point[2]))
    return np.array(relative_coords)

cones = np.array([
    [100 + 30 * 0, -5.25, -1], [100 + 30 * 1, -5.25, 1],
    [100 + 30 * 2, -5.25, -1], [100 + 30 * 3, -5.25, 1],
    [100 + 30 * 4, -5.25, -1], [100 + 30 * 5, -5.25, 1],
    [100 + 30 * 6, -5.25, -1], [100 + 30 * 7, -5.25, 1],
    [100 + 30 * 8, -5.25, -1], [100 + 30 * 9, -5.25, 1],
])

x, y, yaw = 0, 0, 0
a= to_relative_coordinates(x, y, yaw, cones)
for idx, (conex, coney, conez) in enumerate(cones):
    print(idx)