import numpy as np

# -------------------------------
# Define some functions to match current position with reference path
# -------------------------------

# Find closest point on the reference path compared witch current position
def find_closest_point(pose, reference_path, start_index):
    # x and y distance from current position (pose) to every point in 
    # the reference path starting at a certain starting index
    xlist = reference_path['x'][start_index:] - pose[0]
    ylist = reference_path['y'][start_index:] - pose[1]
    # Index of closest point by Pythagoras theorem
    index_closest = start_index+np.argmin(np.sqrt(xlist*xlist + ylist*ylist))
    print('find_closest_point results in', index_closest)
    return index_closest

# Return the point on the reference path that is located at a certain distance 
# from the current position
def index_last_point_fun(start_index, wp, dist):
    pathpoints = wp.shape[1]
    # Cumulative distance covered
    cum_dist = 0
    # Start looping the index from start_index to end
    for i in range(start_index, pathpoints-1):
        # Update comulative distance covered
        cum_dist += np.linalg.norm(wp[:,i] - wp[:,i+1])
        # Are we there yet?
        if cum_dist >= dist:
            return i + 1
    # Desired distance was never covered, -1 for zero-based index
    return pathpoints - 1

# Create a list of N waypoints
def get_current_waypoints(start_index, wp, N, dist):
    # Determine index at reference path that is dist away from starting point
    last_index = index_last_point_fun(start_index, wp, dist)
    # Calculate amount of indices between last and start point
    delta_index = last_index - start_index
    # Dependent on the amount of indices, do
    if delta_index >= N: 
        # There are more than N path points available, so take the first N ones
        index_list = list(range(start_index, start_index+N+1))
        print('index list with >= N points:', index_list)
    else:
        # There are less than N path points available, so add the final one multiple times
        index_list = list(range(start_index, last_index)) + [last_index]*(N-delta_index+1)
        print('index list with < N points:', index_list)
    return wp[:,index_list]