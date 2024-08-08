# Autonomous Drone Tracking System

This project is focused on developing an algorithm that enables a drone to autonomously track a user along a designated route, while dynamically detecting and avoiding obstacles. The project integrates computer vision, data processing methodologies, and advanced motion planning algorithms to ensure safe and efficient navigation.

## Project Overview

### Purpose
The primary goal of this project is to create an autonomous drone tracking system that identifies a user's initial position and destination, and continuously generates an optimal path connecting these points. The algorithm is designed to detect obstacles in real-time and compute the shortest and safest trajectory to the destination, avoiding any obstacles encountered along the way.

### Key Features
- **Autonomous Tracking Algorithm**: Developed an algorithm that integrates computer vision, data processing, and motion planning techniques to enable autonomous tracking.
- **Real-time Detection**: Utilized the YOLOv8 algorithm for real-time detection of the user and obstacles in the environment.
- **Motion Planning**: Implemented the Rapidly Exploring Random Tree Star (RRT*) algorithm to continuously generate an optimal path considering dynamic obstacles.
- **Visual Feedback**: Displayed identified obstacles and the computed optimal path on the user's laptop screen for guidance during navigation.

### Technical Details
- **Computer Vision**: Frames from the drone's camera are processed using OpenCV to detect the user. The YOLOv8 algorithm is used to identify both static and dynamic obstacles.
- **Feedback Loop**: A feedback loop calculates the user's position and error, guiding the drone's tracking behavior.
- **Path Planning**: The RRT* algorithm processes the user's binary contour and YOLO's output to identify obstacles and compute the optimal path from start to destination.
- **User Interface**: The computed RRT* graph and detected obstacles are displayed on the laptop screen, providing real-time navigation guidance.

## Project Deliverables
1. **Autonomous Tracking Algorithm**: Integration of computer vision techniques, data processing, and motion planning algorithms.
2. **Real-time Obstacle Detection**: Using the YOLOv8 algorithm to detect users and obstacles in the environment.
3. **Optimal Path Generation**: Utilizing the RRT* algorithm to generate an optimal path from the user's current position to the destination.
4. **Visual Display**: Real-time display of identified obstacles and computed optimal paths on the user's laptop screen.

## Running the Code

### Prerequisites
- **Python**: Ensure Python is installed on your system.
- **Dependencies**: Install necessary libraries using the `requirements.txt` file provided.

### Steps to Run the Project

1. **Download the YOLO Dataset**:
    - Run the `load_yolo.py` file to download the dataset responsible for detecting objects.
    - This step is essential for enabling the YOLOv8 algorithm to accurately identify obstacles.

    ```bash
    python load_yolo.py
    ```

2. **Connect to the Drone**:
    - Ensure your laptop is connected to the drone via Wi-Fi.

3. **Run the Autonomous Tracking Script**:
    - Execute the `optimized_track.py` file to start the drone's autonomous tracking and obstacle avoidance.

    ```bash
    python optimized_track.py
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any inquiries, please contact [Maher Morsi](mailto:maher.morsi@gmail.com).
