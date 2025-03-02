import cv2
import numpy as np
import time
from robot_framework import RobotActionGenerator

class SimpleFlightSimEnv: # but is this really a flight sim?
    def __init__(self, width=640, height=640):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        self.x = self.width // 2
        self.y = self.height // 2
        self.angle = 0
        return self.render()
    
    def step(self, action):
        scale = 5.0
        dx = np.clip(action[0], -1, 1) * scale
        dy = np.clip(action[1], -1, 1) * scale
        self.x += dx
        self.y += dy
        self.x = np.clip(self.x, 0, self.width)
        self.y = np.clip(self.y, 0, self.height)
        done = False
        return self.render(), done
    
    def render(self):
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        pts = np.array([[self.x, self.y - 10], [self.x - 5, self.y + 5], [self.x + 5, self.y + 5]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], (0, 0, 255))
        return frame

def main():
    env = SimpleFlightSimEnv()
    frame = env.reset()
    robot_controller = RobotActionGenerator(yolo_model_size="n") # n, s, m, whatever else
    
    while True:
        action, viz_frame = robot_controller.process_frame(frame)
        frame, _ = env.step(action)
        cv2.imshow("Flight Sim", frame)
        cv2.imshow("Robot Viz", viz_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
