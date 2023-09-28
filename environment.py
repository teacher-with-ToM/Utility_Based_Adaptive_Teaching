from __future__ import annotations

from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_COLOR
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

import numpy as np

class MultiGoalsEnv(MiniGridEnv):
    def __init__(
        self,
        agent_goal: int,
        agent_view_size: int,
        size=20,
        agent_start_pos: tuple=(1, 1),
        agent_start_dir: int=0,
        num_colors: int=4,
        max_steps: int | None = None,
        **kwargs,
    ):  
        self.agent_goal = agent_goal
        self.num_doors = num_colors
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_view_size = agent_view_size

        if self.agent_view_size >= size:
            self.see_through_walls = True
        else:
            self.see_through_walls = False

        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            self.max_steps = size**2
        else:
            self.max_steps = max_steps

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            agent_view_size=agent_view_size,
            max_steps=self.max_steps,
            see_through_walls=self.see_through_walls,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Open the door with the right color"

    def _gen_grid(self, width: int, height: int):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        self.obj_idx = [self.agent_start_pos]

        # Place walls around
        for i in range(0, height):
            for j in range(0, width):
                self.grid.set(0, i, Wall())
                self.grid.set(width - 1, i, Wall())
                self.grid.set(j, 0, Wall())
                self.grid.set(j, height - 1, Wall())

        # Place the doors and keys
        self.doors = []
        self.keys = []
        for ii in range(self.num_doors):

            # Create door and key at random position
            i_door, i_key = np.random.randint(1, self.width - 1, size=2)
            j_door, j_key = np.random.randint(1, self.height - 1, size=2)

            # Ensure no other object at the position
            while (i_door, j_door) in self.obj_idx:
                i_door = np.random.randint(1, self.width - 1)
                j_door = np.random.randint(1, self.height - 1)
            self.obj_idx.append((i_door, j_door))
            while (i_key, j_key) in self.obj_idx:
                i_key = np.random.randint(1, self.width - 1)
                j_key = np.random.randint(1, self.height - 1)
            self.obj_idx.append((i_key, j_key))

            door = Door(IDX_TO_COLOR[ii+1], is_locked=True)
            key = Key(IDX_TO_COLOR[ii+1])
            self.doors.append(door)
            self.keys.append(key)
            # Add door and key to the env
            self.grid.set(i_door, j_door, door)
            self.grid.set(i_key, j_key, key)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Open the door with the right color"

    def reset_grid(self):

        if self.agent_view_size >= self.height:
            self.see_through_walls = True
        else:
            self.see_through_walls = False

        self.carrying = None
        self.step_count = 0

        self.grid = Grid(self.width, self.height)

        # Place walls around
        for i in range(0, self.height):
            for j in range(0, self.width):
                self.grid.set(0, i, Wall())
                self.grid.set(self.width - 1, i, Wall())
                self.grid.set(j, 0, Wall())
                self.grid.set(j, self.height - 1, Wall())
        
        self.doors = []
        self.keys = []
        for ii in range(self.num_doors):

            # Create door and key at random position
            i_door, j_door = self.obj_idx[1 + 2 * ii]
            i_key, j_key = self.obj_idx[1 + 2 * ii + 1]

            door = Door(IDX_TO_COLOR[ii+1], is_locked=True)
            key = Key(IDX_TO_COLOR[ii+1])
            self.doors.append(door)
            self.keys.append(key)

            # Add door and key to the env
            self.grid.set(i_door, j_door, door)
            self.grid.set(i_key, j_key, key)

        self.reset_agent_pos()

    def reset_agent_pos(self):
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def step(self, action: Actions):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.doors[self.agent_goal - 1].is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
    
##
# Complex environment for demonstration
##

class MultiRoomsGoalsEnv(MiniGridEnv):
    def __init__(
        self,
        agent_goal: int,
        agent_view_size: int,
        size=20,
        agent_start_pos: tuple=(1, 1),
        agent_start_dir: int=0,
        num_colors: int=4,
        num_rooms: int=3,
        max_steps: int | None = None,
        **kwargs,
    ):  
        self.agent_goal = agent_goal
        self.num_doors = num_colors
        self.num_rooms = num_rooms
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_view_size = agent_view_size

        if self.agent_view_size >= size:
            self.see_through_walls = True
        else:
            self.see_through_walls = False

        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            self.max_steps = int(size**2 / 2)
        else:
            self.max_steps = max_steps

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            agent_view_size=agent_view_size,
            max_steps=self.max_steps,
            see_through_walls=self.see_through_walls,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Open the door with the right color"

    def _gen_grid(self, width: int, height: int):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        self.obj_idx = [self.agent_start_pos]

        self.wall_idx = []
        # Place walls around
        for i in range(0, height):
            for j in range(0, width):
                self.grid.set(0, i, Wall())
                self.grid.set(width - 1, i, Wall())
                self.grid.set(j, 0, Wall())
                self.grid.set(j, height - 1, Wall())

        # Create rooms
        room_size = self.height // self.num_rooms
        opening_idx = [((rr - 1) * room_size + room_size // 2) for rr in range(1, self.num_doors)]
        for rr in range(1, self.num_rooms):
            for i in range(0, height):
                if i not in opening_idx:
                    self.grid.set(rr * room_size, i, Wall())
                    self.wall_idx.append((rr * room_size, i))

                    self.grid.set(i, rr * room_size, Wall())
                    self.wall_idx.append((i, rr * room_size))
        
        mid_idx = self.num_rooms // 2
        first_room_idx = [(i, j) for i in range(mid_idx * room_size, (mid_idx + 1) * room_size) \
                                for j in range((self.num_rooms - 1) * room_size, self.width)]
        
        # Place the doors and keys
        self.doors = []
        self.keys = []
        for ii in range(self.num_doors):

            # Create door and key at random position
            i_door, i_key = np.random.randint(1, self.width - 1, size=2)
            j_door, j_key = np.random.randint(1, self.height - 1, size=2)

            # Ensure no other object at the position (and not in the first room)
            while ((i_door, j_door) in self.obj_idx) or ((i_door, j_door) in self.wall_idx) or ((i_door, j_door) in first_room_idx):
                i_door = np.random.randint(1, self.width - 1)
                j_door = np.random.randint(1, self.height - 1)
            self.obj_idx.append((i_door, j_door))
            while ((i_key, j_key) in self.obj_idx or ((i_key, j_key) in self.wall_idx)) or ((i_key, j_key) in first_room_idx):
                i_key = np.random.randint(1, self.width - 1)
                j_key = np.random.randint(1, self.height - 1)
            self.obj_idx.append((i_key, j_key))

            door = Door(IDX_TO_COLOR[ii+1], is_locked=True)
            key = Key(IDX_TO_COLOR[ii+1])
            self.doors.append(door)
            self.keys.append(key)
            # Add door and key to the env
            self.grid.set(i_door, j_door, door)
            self.grid.set(i_key, j_key, key)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Open the door with the right color"

    def reset_grid(self):

        if self.agent_view_size >= self.height:
            self.see_through_walls = True
        else:
            self.see_through_walls = False

        self.carrying = None
        self.step_count = 0

        self.grid = Grid(self.width, self.height)

        # Place walls around
        for i in range(0, self.height):
            for j in range(0, self.width):
                self.grid.set(0, i, Wall())
                self.grid.set(self.width - 1, i, Wall())
                self.grid.set(j, 0, Wall())
                self.grid.set(j, self.height - 1, Wall())

        # Place walls
        for (i,j) in self.wall_idx:
            self.grid.set(i, j, Wall())
        
        self.doors = []
        self.keys = []
        for ii in range(self.num_doors):

            # Place doors and keys
            i_door, j_door = self.obj_idx[1 + 2 * ii]
            i_key, j_key = self.obj_idx[1 + 2 * ii + 1]

            door = Door(IDX_TO_COLOR[ii+1], is_locked=True)
            key = Key(IDX_TO_COLOR[ii+1])
            self.doors.append(door)
            self.keys.append(key)

            self.grid.set(i_door, j_door, door)
            self.grid.set(i_key, j_key, key)

        self.reset_agent_pos()

    def reset_agent_pos(self):
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def step(self, action: Actions):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.doors[self.agent_goal-1].is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info