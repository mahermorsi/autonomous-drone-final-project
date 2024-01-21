import numpy as np
import matplotlib.pyplot as plt
import random
import math
from PIL import Image


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


def rrt_star(start, goal, img, max_iter=1000, delta=10, radius=10):
    img_height, img_width = img.shape
    nodes = [start]

    for _ in range(max_iter):
        random_point = Node(
            random.randint(0, img_width - 1), random.randint(0, img_height - 1)
        )

        nearest_node = min(
            nodes,
            key=lambda n: math.sqrt(
                (n.x - random_point.x) ** 2 + (n.y - random_point.y) ** 2
            ),
        )

        new_point = Node(nearest_node.x, nearest_node.y)
        angle = math.atan2(
            random_point.y - nearest_node.y, random_point.x - nearest_node.x
        )
        new_point.x += int(delta * math.cos(angle))
        new_point.y += int(delta * math.sin(angle))

        # Ensure the new point is within the image bounds
        new_point.x = max(0, min(img_width - 1, new_point.x))
        new_point.y = max(0, min(img_height - 1, new_point.y))

        if img[new_point.y, new_point.x] == 0:
            continue  # Skip if the new point is in an obstacle

        near_nodes = [
            node
            for node in nodes
            if math.sqrt((node.x - new_point.x) ** 2 + (node.y - new_point.y) ** 2)
            < radius
        ]

        min_cost_node = nearest_node
        min_cost = nearest_node.cost + math.sqrt(
            (nearest_node.x - new_point.x) ** 2 + (nearest_node.y - new_point.y) ** 2
        )

        for node in near_nodes:
            cost = node.cost + math.sqrt(
                (node.x - new_point.x) ** 2 + (node.y - new_point.y) ** 2
            )
            if cost < min_cost and img[node.y, node.x] != 0:
                min_cost_node = node
                min_cost = cost

        new_point.parent = min_cost_node
        new_point.cost = min_cost

        nodes.append(new_point)

        for node in near_nodes:
            cost = new_point.cost + math.sqrt(
                (node.x - new_point.x) ** 2 + (node.y - new_point.y) ** 2
            )
            if cost < node.cost and img[node.y, node.x] != 0:
                node.parent = new_point
                node.cost = cost

    path = []
    current_node = min(
        nodes, key=lambda n: math.sqrt((n.x - goal.x) ** 2 + (n.y - goal.y) ** 2)
    )
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
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], "g.-")
    plt.title("RRT* Path Planning")
    plt.savefig("final path.png")
    plt.show()


def find_rrt_path(imagePath):
    global clicked_points
    clicked_points = []

    # Load the image (black and white, where black represents obstacles and white represents the path)
    image = Image.open(imagePath).convert("1")
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
    path_result = rrt_star(start, goal, obstacles)
    if path_result:
        plot_path(obstacles, path_result, (start.x, start.y), (goal.x, goal.y))
    else:
        print("path was not found!")
