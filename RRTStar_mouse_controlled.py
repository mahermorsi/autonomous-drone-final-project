import math
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


def calculate_new_and_optimal_points(nearest_node, random_point, goal, nearest_node_to_goal_point, delta, img_width, img_height):
    new_point = Node(nearest_node.x, nearest_node.y)
    optimal_point = Node(nearest_node.x, nearest_node.y)

    angle = math.atan2(random_point.y - nearest_node.y, random_point.x - nearest_node.x)
    angle_goal = math.atan2(goal.y - nearest_node_to_goal_point.y, goal.x - nearest_node_to_goal_point.x)

    new_point.x += int(delta * math.cos(angle))
    new_point.y += int(delta * math.sin(angle))

    optimal_point.x += int(delta * math.cos(angle_goal))
    optimal_point.y += int(delta * math.sin(angle_goal))

    # Ensure the new point and the optimal point is within the image bounds
    new_point.x = max(0, min(img_width - 1, new_point.x))
    new_point.y = max(0, min(img_height - 1, new_point.y))

    optimal_point.x = max(0, min(img_width - 1, optimal_point.x))
    optimal_point.y = max(0, min(img_height - 1, optimal_point.y))

    return new_point, optimal_point


def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def rrt_star(start, goal, img, max_iter=2000, delta=10, radius=10):
    img_height, img_width = img.shape
    nodes = [start]

    for _ in range(max_iter):
        random_point = Node(random.randint(0, img_width - 1), random.randint(0, img_height - 1))

        nearest_node = min(nodes, key=lambda n: calculate_distance(n, random_point),)
        nearest_node_to_goal_point = min(nodes, key=lambda n: calculate_distance(n,goal),)

        new_point, optimal_point = calculate_new_and_optimal_points(nearest_node, random_point, goal, nearest_node_to_goal_point, delta, img_width, img_height)

        if img[new_point.y, new_point.x] == 0:
            continue  # Skip if the new point is in an obstacle

        # if img[optimal_point.y, optimal_point.x] != 0:
        #     new_point.x = optimal_point.x
        #     new_point.y = optimal_point.y

        near_nodes = [node for node in nodes if calculate_distance(node, new_point) <= radius]

        min_cost_node = nearest_node
        min_cost = nearest_node.cost + calculate_distance(nearest_node,new_point)

        for node in near_nodes:
            cost = node.cost + calculate_distance(node, new_point)

            if cost < min_cost and img[node.y, node.x] != 0:
                min_cost_node = node
                min_cost = cost

        new_point.parent = min_cost_node
        new_point.cost = min_cost

        nodes.append(new_point)

        for node in near_nodes:
            cost = new_point.cost + calculate_distance(node, new_point)
            if cost < node.cost and img[node.y, node.x] != 0:
                node.parent = new_point
                node.cost = cost

    path = []
    current_node = min(nodes, key=lambda n: calculate_distance(n, goal))
    while current_node:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent

    return path[::-1]  # Reverse the path to start from the start node


def on_mouse_click(event):
    if event.dblclick:
        global clicked_points
        clicked_points.append((int(event.xdata), int(event.ydata)))
        if len(clicked_points) == 2:
            plt.close()


def plot_path(image, path, start, goal):
    plt.imshow(image, cmap="gray")
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], "g.-")
    plt.plot(start[0], start[1], 'bo', label='Start')
    plt.plot(goal[0], goal[1], 'go', label='Goal')
    plt.title("RRT* Path Planning")
    plt.savefig("final path.png")
    plt.show()


def find_rrt_path(image_path):
    global clicked_points
    clicked_points = []

    # Load the image (black and white, where black represents obstacles and white represents the path)
    image = Image.open(image_path).convert("1")
    image_array = np.array(image)
    obstacles = np.where(image_array == 0, 0, 1)

    # Display the image for the user to select start and goal points
    fig, ax = plt.subplots()
    ax.imshow(obstacles, cmap="gray")
    plt.title("Select Start and Goal Points")
    plt.tight_layout()

    cid = fig.canvas.mpl_connect("button_press_event", on_mouse_click)

    plt.show()

    # Disconnect the callback after the window is closed
    fig.canvas.mpl_disconnect(cid)

    while len(clicked_points) < 2:
        plt.pause(0.1)

    # Extract start and goal points
    start = Node(clicked_points[0][0], clicked_points[0][1])
    goal = Node(clicked_points[1][0], clicked_points[1][1])

    #Calculate ideal path
    path_result = rrt_star(start, goal, obstacles)
    if path_result:
        plot_path(obstacles, path_result, (start.x, start.y), (goal.x, goal.y))
    else:
        print("path was not found!")


find_rrt_path('example screenshots/masked_image.png')
