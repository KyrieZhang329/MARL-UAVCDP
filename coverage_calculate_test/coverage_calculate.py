import numpy as np
import time


class DroneCoverage:
    def __init__(self, drone1_pos, drone2_pos, drone1_scan_range,drone2_scan_range,drone1_speed, drone2_speed):

        self.drone1_pos = np.array(drone1_pos)
        self.drone2_pos = np.array(drone2_pos)
        self.drone1_scan_range = drone1_scan_range
        self.drone2_scan_range = drone2_scan_range
        self.scanned_points = set()
        

        self.add_scanned_points(self.drone1_scan_range, self.drone2_scan_range, 
                                self.drone1_pos, self.drone2_pos)
        

        self.drone1_speed = drone1_speed
        self.drone2_speed = drone2_speed
    
    def add_scanned_points(self, drone1_scan_range, drone2_scan_range, drone1_pos, drone2_pos):

        points1 = self.scan_points(drone1_scan_range, drone1_pos)
        self.scanned_points.update(points1)

        points2 = self.scan_points(drone2_scan_range, drone2_pos)
        self.scanned_points.update(points2)

    def scan_points(self, scan_range, pos):
        x, y, z = pos
        scanned_points_set = set()
        
        for dx in range(int(-scan_range), int(scan_range) + 1):
            for dy in range(int(-scan_range), int(scan_range) + 1):
                for dz in range(int(-scan_range), int(scan_range) + 1):
                    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if distance <= scan_range:
                        scanning_point = (int(x + dx), int(y + dy), int(z + dz))
                        scanned_points_set.add(scanning_point)
        
        return scanned_points_set

    def orient(self):
        direction = np.random.uniform(-1, 1, 3)
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction /= norm
        return direction

    def move_drones(self):
        direction1 = self.orient()
        self.drone1_pos = self.drone1_pos + direction1 * self.drone1_speed
        direction2 = self.orient()
        self.drone2_pos = self.drone2_pos + direction2 * self.drone2_speed
        self.add_scanned_points(self.drone1_scan_range, self.drone2_scan_range, self.drone1_pos, self.drone2_pos)

    def simulate(self, steps, total_count):
        print(f"开始模拟飞行 {steps} 步")
        print(f"每10步后停顿2秒")
        for i in range(steps):
            self.move_drones()
            coverage_percent, scanned_points = self.calculate_coverage(total_count)
            print(f"第 {i+1} 步飞行，已扫描{scanned_points}个点，覆盖率更新为{coverage_percent*100}%")
            pause_duration = 0.5
            time.sleep(pause_duration)

    
    def calculate_coverage(self, total_count):
        scanned_count = len(self.scanned_points)
        coverage_percent = round(min(scanned_count / total_count, 1.00), 4)
        return coverage_percent, scanned_count
