import threading
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random


class RRT:
    def __init__(self, start, goal, obstacles, image_size, step_size=90, max_iterations=2000):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.image_size = image_size
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tree = {start: None}

    def generate_random_point(self):
        x = random.randint(0, self.image_size[0] - 1)
        y = random.randint(0, self.image_size[1] - 1)
        return (x, y)

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
            return (x, y)

    def is_collision_free(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        x = np.linspace(x1, x2, num=100, dtype=int)
        y = np.linspace(y1, y2, num=100, dtype=int)
        x = np.clip(x, 0, self.image_size[1] - 1)
        y = np.clip(y, 0, self.image_size[0] - 1)
        for i in range(len(x)):
            if np.all(self.obstacles[y[i], x[i]] == 0):  # Check if any obstacle is present at coordinates
                return False
        return True

    def find_path(self):
        fig, ax = plt.subplots()
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
                ax.clear()
                path=self.construct_path()
                ax.imshow(self.obstacles, cmap='gray')
                ax.plot([point[0] for point in path], [point[1] for point in path], 'r-')
                ax.plot(self.start[0], self.start[1], 'go', markersize=5)
                ax.plot(self.goal[0], self.goal[1], 'bo', markersize=5)
                ax.set_title("RRT Path Planning (Iteration: {})".format(_ + 1))
                plt.pause(0.01)
                return path

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


def plot_image(obstacles):
    plt.imshow(obstacles, cmap='gray')
    plt.show()


def find_RRT_path(objects_image_path):
    # Load image

    # start = get_start_coordination.find_yellow_pixel(path_of_yellow_img)
    # image_path = white_mask.filter_colors(objects_image_path)
    image = Image.open(objects_image_path).convert('1')  # Convert to black and white
    image_array = np.array(image)
    obstacles = np.where(image_array == 0, 0, 1)  # 0 represents obstacles, 1 represents path

    plot_thread = threading.Thread(target=plot_image, args=(obstacles,))
    plot_thread.daemon = True
    plot_thread.start()
    start_x = int(input("Enter start x-coordinate: "))
    start_y = int(input("Enter start y-coordinate: "))
    start = (start_x, start_y)
    plt.plot(start_x, start_y, 'go', markersize=5, label='Start')

    end_x = int(input("Enter end x-coordinate: "))
    end_y = int(input("Enter end y-coordinate: "))
    plt.plot(end_x, end_y, 'bo', markersize=5, label='Goal')
    goal = (end_x, end_y)

    # print(image_array.shape)

    # Create RRT planner
    planner = RRT(start, goal, obstacles, image_array.shape)

    # Find path
    path = planner.find_path()

    if path:
        print("Path found!")
        # Visualize path
        plt.imshow(image_array, cmap='gray')
        plt.plot([point[0] for point in path], [point[1] for point in path], 'r-')
        plt.plot(start[0], start[1], 'go', markersize=5)
        plt.plot(goal[0], goal[1], 'bo', markersize=5)
        plt.title("RRT Path Planning")
        plt.savefig("final_path.png")
        plt.show()

    else:
        print("Failed to find a path.")



