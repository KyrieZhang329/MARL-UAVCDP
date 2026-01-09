import numpy as np
from marl_uav.env.core import World,Agent,Landmark
from marl_uav.env.config import DRONE_CONFIGS,MAP_SIZE
from marl_uav.env.cover_scan import GridMapScan

class Scenario:
    def make_world(self):
        world = World()
        world.dim_c = 2
        world.dim_p = 2
        formation = ['SCOUT','SCOUT','RELAY','EXECUTOR']

        world.agents = []
        for i,uav_type in enumerate(formation):
            agent = Agent(uav_type=uav_type)
            agent.name = f'uva_{i}_{uav_type}'
            agent.collide = True
            agent.silent = False

            agent.state.p_pos = np.random.uniform(-MAP_SIZE/2,MAP_SIZE/2,world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.layer = 1.0
            agent.state.update_height()
            world.agents.append(agent)

        self.scanner = GridMapScan(map_size=MAP_SIZE)
        return world

    def reset_world(self,world):
        self.scanner.reset()
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-MAP_SIZE/2,MAP_SIZE/2,world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.accumulated_fatigue = 0.0
            agent.is_weak_battery = False
            agent.state.layer = np.random.uniform(0.5,2.0)
            agent.state.update_height()

    def reward(self,world,agent):
        reward = 0
        r=agent.get_sensing_radius()
        new_cells = self.scanner.update_coverage(agent.state.p_pos,r)
        reward += new_cells*1.0
        reward *= agent.reward_weight
        if agent.is_weak_battery:
            action_mag = np.linalg.norm(agent.action.u)
            penalty = 0.01*action_mag/agent.battery_capacity
            reward -= penalty
        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                distance_2d = np.linalg.norm(a.state.p_pos - agent.state.p_pos)
                distance_z = abs(a.state.height-agent.state.height)
                min_distance_2d = agent.size+a.size
                min_distance_z = 2.0
                if distance_2d < min_distance_2d and distance_z < min_distance_z:
                    reward -= 10.0
                    if not hasattr(world,'collisions'):world.collisions=0
                    world.collisions += 1
        return reward

    def get_comm(self, world):
        adj = np.zeros((len(world.agents), len(world.agents)))
        for i,agent_a in enumerate(world.agents):
            for j,agent_b in enumerate(world.agents):
                if i == j:
                    continue
                distance = np.linalg.norm(agent_a.state.p_pos - agent_b.state.p_pos)
                if distance < agent_a.max_comm:
                    adj[i][j] = 1
        return adj

    def observation(self,agent,world):
        self.state = [
            agent.state.p_vel[0]/agent.max_speed,
            agent.state.p_vel[1]/agent.max_speed,
            agent.state.p_pos[0]/(MAP_SIZE/2),
            agent.state.p_pos[1]/(MAP_SIZE/2),
            1.0 if agent.is_weak_battery else 0.0,
            agent.state.height,
        ]

        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos-agent.state.p_pos)
        entity_obs = np.concatenate(entity_pos) if entity_pos else np.array([])
        other_obs = []
        for other_agent in world.agents:
            if other_agent is agent:
                continue
            distance = np.linalg.norm(other_agent.state.p_pos-agent.state.p_pos)
            is_visible = (distance<=agent.max_comm)
            if is_visible:
                rel_pos = other_agent.state.p_pos-agent.state.p_pos
                other_obs.extend([rel_pos[0],rel_pos[1],1.0])
            else:
                other_obs.extend([0.0,0.0,0.0])
        return np.concatenate([
            np.array(self.state),
            entity_obs,
            np.array(other_obs),
        ])