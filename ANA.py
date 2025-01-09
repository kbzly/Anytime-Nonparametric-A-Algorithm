import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, \
    joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
from queue import PriorityQueue
### YOUR IMPORTS HERE ###

def angle_change(angle):
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle

def action_cost(current, neighbor):
    dx = current[0] - neighbor[0]
    dy = current[1] - neighbor[1]
    dtheta = min(abs(current[2] - neighbor[2]), 2 * np.pi - abs(current[2] - neighbor[2]))
    return np.sqrt(dx ** 2 + dy ** 2 + dtheta ** 2)

def heuristic(node, goal):
    dx = node[0] - goal[0]
    dy = node[1] - goal[1]
    dtheta = min(abs(node[2] - goal[2]), 2 * np.pi - abs(node[2] - goal[2]))
    return np.sqrt(dx ** 2 + dy ** 2 + dtheta ** 2)

def weighted_heuristic(node, goal, linear_weight=1.0, angular_weight=0.5):
    dx = node[0] - goal[0]
    dy = node[1] - goal[1]
    dtheta = min(abs(node[2] - goal[2]), 2 * np.pi - abs(node[2] - goal[2]))
    linear_distance = np.sqrt(dx ** 2 + dy ** 2)
    return linear_weight * linear_distance + angular_weight * dtheta

def close_to_goal(current_state, goal):
    (x1, y1, theta1) = current_state
    (x2, y2, theta2) = goal
    tolerance = 1e-4
    distance_1 = np.abs(x1 - x2)
    distance_2 = np.abs(y1 - y2)
    theta_diff = abs(angle_change(theta1-theta2))
    if distance_1 < tolerance and distance_2 < tolerance and theta_diff < tolerance:
        return True
    else:
        return False

def backtrack(closed_list, current_node):
    path = []
    while current_node is not None:
        path.append(current_node)
        current_node = closed_list.get(current_node)
    return path[::-1]


def get_neighbors(node, mode="8-connected"):
    dx, dy, dtheta = 0.1, 0.1, np.pi / 2
    if mode == "4-connected":
        moves = [(dx, 0, 0), (0, dy, 0), (-dx, 0, 0), (0, -dy, 0), (0, 0, dtheta), (0, 0, -dtheta)]
    if mode == "8-connected":
        moves = [(dx, 0, 0), (dx, 0, dtheta), (dx, 0, -dtheta), (0, dy, 0), (0, dy, dtheta), (0, dy, -dtheta),
                 (-dx, 0, 0), (-dx, 0, dtheta), (-dx, 0, -dtheta), (0, -dy, 0), (0, -dy, dtheta), (0, -dy, -dtheta),
                 (0, 0, dtheta), (0, 0, -dtheta), (dx, dy, 0), (dx, dy, dtheta), (dx, dy, -dtheta),
                 (-dx, dy, 0), (-dx, dy, dtheta), (-dx, dy, -dtheta), (dx, -dy, 0), (dx, -dy, dtheta),
                 (dx, -dy, -dtheta),
                 (-dx, -dy, 0), (-dx, -dy, dtheta), (-dx, -dy, -dtheta)]  # diagonal actions
    return [(node[0] + move[0], node[1] + move[1], angle_change(node[2] + move[2])) for move in moves]


def astar(start, goal, collision_fn, mode="8-connected"):
    open_list = PriorityQueue()
    open_list.put((0, start)) # put the start node in the quene
    gcosts = {start: 0}
    fcosts = {start: heuristic(start, goal)}
    # closed_list contains all node passed, they will not be passed again.
    close_list = set()
    closed_list = {} # It is a map, key is the state current passed, value is its father's state

    while not open_list.empty():
        _, current_node = open_list.get()

        if current_node in close_list:
            continue

        if collision_fn(current_node):
            continue

        close_list.add(current_node)

        if close_to_goal(current_node, goal):
            return gcosts[current_node], backtrack(closed_list,current_node)

        # generate neighbors, neighbors is a list of node
        for neighbor in get_neighbors(current_node):
            cost_so_far = gcosts[current_node] + action_cost(current_node, neighbor)

            if neighbor in close_list or (neighbor in gcosts and cost_so_far >= gcosts[neighbor]):
                continue

            closed_list[neighbor] = current_node
            gcosts[neighbor] = cost_so_far
            fcosts[neighbor] = cost_so_far + heuristic(neighbor, goal)
            open_list.put((fcosts[neighbor], neighbor))

    print("No Solution Found.")
    return None


def ANAstar(start, goal, collision_fn, time_limit=10):
    start_time = time.time()  # Record the start time
    G = 1000
    E = float('inf')
    open_list = PriorityQueue()
    gcosts = {start: 0}
    closed_list = {}
    cost_over_time = []

    def e_value(node):
        # If node is found in gcosts, return the associated cost. If node is not found in gcosts, return float('inf')
        return (G - gcosts.get(node, float('inf'))) / heuristic(node, goal)

    # negating the priority values, pop out the maximun of original number
    open_list.put((-e_value(start), start))

    def improve_solution():
        nonlocal G, E
        close_list = set()
        cost_over_time = []

        while not open_list.empty():
            _, current = open_list.get()

            if current in close_list:
                continue

            if collision_fn(current):
                continue

            close_list.add(current)

            if e_value(current) < E:
                E = e_value(current)

            if close_to_goal(current, goal):
                G = gcosts[current]
                return current, closed_list

            for neighbor in get_neighbors(current):
                new_cost = gcosts[current] + action_cost(current, neighbor)
                if new_cost < gcosts.get(neighbor, float('inf')):
                    # update or add neighbor
                    gcosts[neighbor] = new_cost
                    # recode parent
                    closed_list[neighbor] = current
                    if new_cost + heuristic(neighbor, goal) < G:
                        open_list.put((-e_value(neighbor), neighbor))
        return None, None

    while not open_list.empty():
        if time.time() - start_time > time_limit:  # Check if the time limit is exceeded
            break

        result = improve_solution()
        cost_over_time.append((time.time() - start_time, G))
        # in most siuation open_list run out at last iteration without closed to goal, def improve_solution() give None
        if result == (None, None):
            break
        current, closed_list = result
        path_suboptimal = backtrack(closed_list, current)

        # G changed, some members in open list will be not qualified, g(s)+h(s)<G
        list_for_update = PriorityQueue()
        while not open_list.empty():
            _, node = open_list.get()
            if gcosts[node] + heuristic(node, goal) < G:
                list_for_update.put((e_value(node), node))
        open_list = list_for_update

    return G, path_suboptimal, cost_over_time




#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    # This the environment in HW3
    robots, obstacles = load_env('pr2doorway.json')
    # This the customized environment
    # robots, obstacles = load_env('pr2customized_environment.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))

    start_config = tuple(get_joint_positions(robots['pr2'], base_joints)) # (-3.4, -1.4, 0.05)
    goal_config = (2.6, -1.3, -np.pi/2)
    # the goal_config of the second Implementation
    # goal_config = (2.6, 2.4, np.pi / 2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
    cost, path, cost_over_time = ANAstar(start_config, goal_config, collision_fn, 1000)
    # cost, path = astar(start_config, goal_config, collision_fn)
    print("The cost is", cost)
    for i in range(len(path) - 1):
        # draw_sphere_marker((path[i][0], path[i][1], 0.3), 0.05, (0, 0, 0, 1))
        draw_line((*path[i][:2], 0.2), (*path[i + 1][:2], 0.2), 10, (0, 0, 0))
    # draw_line(line_start, line_end, line_width, line_color)
    print(cost_over_time)

    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()