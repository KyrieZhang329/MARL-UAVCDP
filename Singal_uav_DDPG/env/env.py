import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class GridMapScan:
    def __init__(self, map_size, grid_num):
        self.map_size = map_size
        self.grid_num = grid_num
        self.cell_size  = self.map_size/self.grid_num
        self.grid = np.zeros((grid_num, grid_num),dtype=np.int8)


    def trans_coordinates(self,pos):
        offset = self.map_size / 2
        x = np.clip(pos[0], -offset, offset-0.1)
        y = np.clip(pos[1], -offset, offset-0.1)
        col = int((x+offset)/self.cell_size)
        row = int((y+offset)/self.cell_size)
        return col, row

    def update_coverage(self, drone_pos, scan_r):
        r_in_cells = int(scan_r/self.cell_size)+1
        center_row, center_col = self.trans_coordinates(drone_pos)
        newly_covered_count = 0

        for r in range(center_row - r_in_cells, center_row + r_in_cells+1):
            for c in range(center_col - r_in_cells, center_col + r_in_cells+1):
                if 0 <= r < self.grid_num and 0 <= c < self.grid_num:
                    grid_center_x = (c*self.cell_size)-(self.map_size/2)+(self.cell_size/2)
                    grid_center_y = (r*self.cell_size)-(self.map_size/2)+(self.cell_size/2)
                    distance = np.linalg.norm(drone_pos-(grid_center_x, grid_center_y))

                    if distance <= scan_r:
                        if self.grid[r, c] == 0:
                            self.grid[r, c] = 1
                            newly_covered_count += 1
        return newly_covered_count

    def calculate_coverage(self):
        return np.sum(self.grid)/self.grid_num**2


class UAVEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps":10}

    def __init__(self, render_mode=None):
        self.map_size = 20.0
        self.grid_num = 20
        self.max_steps = 100

        self.uav_speed = 1.0
        self.scan_r = 2.0
        self.uav_r = 0.5

        self.obstacles = [
            (5.0, 5.0, 2.0),
            (-5.0, -5.0, 1.5),
            (2.0, -6.0, 1.0)
        ]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.grid_map = GridMapScan(self.map_size, self.grid_num)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid_map.grid.fill(0)
        self.steps = 0
        self.pos = np.random.uniform(-8, 8, size=(2,))
        self.grid_map.update_coverage(self.pos, self.scan_r)
        return self.get_obs(), {}


    def get_obs(self):
        rate = self.grid_map.calculate_coverage()
        return np.array([self.pos[0], self.pos[1], rate], dtype=np.float32)


    def step(self, action):
        self.steps += 1
        move = action*self.uav_speed
        next_pos = self.pos+move
        limit = self.map_size/2
        next_pos = np.clip(next_pos, -limit, limit)

        crash = False
        for ox, oy, orad in self.obstacles:
            distance = np.linalg.norm(next_pos-np.array([ox, oy]))
            if distance <= (self.uav_r+orad):
                crash = True
                break

        if not crash:
            self.pos = next_pos

        new_cells = self.grid_map.update_coverage(next_pos, self.scan_r)
        current_rate = self.grid_map.calculate_coverage()

        reward = 0
        reward += new_cells*1.0

        if crash:
            reward -= 10.0

        terminated = False
        if current_rate > 0.95:
            terminated = True

        truncated = False
        if self.steps >= self.max_steps:
            truncated = True

        return self.get_obs(), reward, terminated, truncated, {}


    def render(self):
        if self.render_mode == "human":
            self.render_frame()


    def render_frame(self):
        if self.screen is None:
            pygame.init()
            self.window_size = 600
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        def to_pixel_x(pos):
            norm = (pos+self.map_size/2)/self.map_size
            px = int(norm*self.window_size)
            return px

        def to_pixel_y(y):
            norm = (y+self.map_size/2)/self.map_size
            px = int(self.window_size-(norm*self.window_size))
            return px

        cell_pixel_size = self.window_size/self.grid_num
        for r in range(self.grid_num):
            for c in range(self.grid_num):
                if self.grid_map.grid[r, c] == 1:
                    px = c*cell_pixel_size
                    py = self.window_size-(r+1)*cell_pixel_size
                    pygame.draw.rect(self.screen, (200, 255, 200), (px, py, cell_pixel_size, cell_pixel_size))

        for ox, oy, orad in self.obstacles:
            center = (to_pixel_x(ox), to_pixel_y(oy))
            rad = int(orad/self.map_size*self.window_size)
            pygame.draw.circle(self.screen, (100, 100, 100), center, rad)

        uav_px = (to_pixel_x(self.pos[0]), to_pixel_y(self.pos[1]))
        uav_r = int(self.uav_r/self.map_size*self.window_size)
        scan_rad_px = int(self.scan_r/self.map_size*self.window_size)

        pygame.draw.circle(self.screen, (255, 0, 0), uav_px, scan_rad_px, 1)
        pygame.draw.circle(self.screen, (255, 0, 0), uav_px, uav_r)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()

