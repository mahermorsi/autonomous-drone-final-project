import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image


class RRT:
    def __init__(self, start, goal, obstacles, image_size, step_size=60, max_iterations=5000):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.image_size = image_size
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tree = {start: None}

    def generate_random_point(self):
        return random.randint(0, self.image_size[0] - 1), random.randint(0, self.image_size[1] - 1)

    def find_nearest_point(self, point):
        distances = [self.euclidean_distance(point, p) for p in self.tree.keys()]
        min_index = np.argmin(distances)
        return list(self.tree.keys())[min_index]

    @staticmethod
    def euclidean_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def step_towards(self, current, target):
        distance = self.euclidean_distance(current, target)
        if distance <= self.step_size:
            return target
        else:
            theta = np.arctan2(target[1] - current[1], target[0] - current[0])
            x = int(current[0] + self.step_size * np.cos(theta))
            y = int(current[1] + self.step_size * np.sin(theta))
            return x, y

    def is_collision_free(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        x = np.linspace(x1, x2, num=100, dtype=int)
        y = np.linspace(y1, y2, num=100, dtype=int)
        x = np.clip(x, 0, self.image_size[1] - 1)
        y = np.clip(y, 0, self.image_size[0] - 1)
        for i in range(len(x)):
            if self.obstacles[y[i], x[i]] == 0:  # Check if any obstacle is present at coordinates
                return False
        return True

    def find_path(self):
        for _ in range(self.max_iterations):
            random_point = self.generate_random_point()
            nearest_point = self.find_nearest_point(random_point)
            new_point = self.step_towards(nearest_point, random_point)

            while not self.is_collision_free(nearest_point, new_point):
                random_point = self.generate_random_point()
                nearest_point = self.find_nearest_point(random_point)
                new_point = self.step_towards(nearest_point, random_point)

            self.tree[new_point] = nearest_point

            if self.euclidean_distance(new_point, self.goal) <= self.step_size:
                self.tree[self.goal] = new_point
                return self.construct_path()

        return None

    def construct_path(self):
        path = []
        current = self.goal
        while current != self.start:
            path.append(current)
            current = self.tree[current]
        path.append(self.start)
        path.reverse()
        return path


def plot_rrt(obstacles, start, goal, path=None):
    fig, ax = plt.subplots()
    ax.imshow(obstacles, cmap='gray')
    ax.plot(start[0], start[1], 'go', markersize=5, label='Start')
    ax.plot(goal[0], goal[1], 'bo', markersize=5, label='Goal')

    if path:
        ax.plot([point[0] for point in path], [point[1] for point in path], 'r-')

    plt.title("RRT Path Planning")
    plt.legend()
    plt.savefig("final_path.png")
    plt.show()
def on_mouse_click(event):
    if event.dblclick:
        global clicked_points
        clicked_points.append((int(event.xdata), int(event.ydata)))
        if len(clicked_points) == 2:
            plt.close()

def find_final_RRT_path(imagePath):
    global clicked_points
    clicked_points = []

    # Load the image (black and white, where black represents obstacles and white represents the path)
    image = Image.open(imagePath).convert('1')
    image_array = np.array(image)
    obstacles = np.where(image_array == 0, 0, 1)

    # Display the image for the user to select start and goal points
    fig, ax = plt.subplots()
    ax.imshow(obstacles, cmap='gray')
    plt.title('Select Start and Goal Points')
    plt.tight_layout()

    cid = fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    plt.show()

    # Disconnect the callback after the window is closed
    fig.canvas.mpl_disconnect(cid)

    while len(clicked_points) < 2:
        plt.pause(0.1)

    # Extract start and goal points
    start = clicked_points[0]
    goal = clicked_points[1]
    print(start)
    print(goal)

    # Create RRT planner
    planner = RRT(start, goal, obstacles, image_array.shape)

    # Find path
    path = planner.find_path()

    if path:
        print("Path found!")
        plot_rrt(obstacles, start, goal, path)
    else:
        print("Failed to find a path.")
