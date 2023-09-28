import numpy as np
from queue import SimpleQueue
from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC, OBJECT_TO_IDX
from PIL import Image

from environment import MultiGoalsEnv, MultiRoomsGoalsEnv
from tools.utils import *

##
# Learner that plans single position transition (action planning)
##

class BayesianLearner:
    def __init__(self,
                 goal_color: int,
                 receptive_field: int,
                 grid_size: int=20,
                 num_colors: int=4,
                 save_render: bool=False,
                 max_steps: int | None = None,
                 env_type: str='MultiGoalsEnv',
                 render_mode: str | None="rgb_array"
                 ) -> None:
        
        self.render_mode = render_mode

        self.env_type = env_type
        self.grid_size = grid_size
        self.goal_color = goal_color + 1
        self.receptive_field = receptive_field
        self.max_steps = max_steps

        # Init with uniform beliefs (no prior)
        self.beliefs = 1 / 4 * np.ones((grid_size, grid_size, 4))

        # Generate feasable environment
        feasible = False
        while not feasible:
            
            feasible = True
            # Generate env
            self.init_env(grid_size=grid_size,
                          num_colors=num_colors)
            self.env.reset()

            # Check feasability
            grid = np.ones((self.env.height, self.env.width))
            for i in range(self.env.height):
                for j in range(self.env.width):
                    grid[i, j] = self.env.grid.get(i,j) is not None

            for obj_idx in self.env.obj_idx:
                grid[obj_idx[0], obj_idx[1]] = 0
                
                path = A_star_algorithm(start=self.env.agent_pos, goal=obj_idx, grid=grid)
                feasible = feasible and (path is not None)
            
                grid[obj_idx[0], obj_idx[1]] = 1
        
        self.reached_subgoal = False

        self.actions = SimpleQueue()
        self.transitions = SimpleQueue()

        self.reward = 0
        self.terminated = False
        self.going_to_subgoal = False
        self.going_to_goal = False

        self.obstacle_grid = np.ones((self.env.height, self.env.width))
        self.shortest_path_subgoal = None
        self.shortest_path_goal = None

        # For rendering
        self.save_render = save_render
        self.render_frames = []
        self.render_beliefs = []

        self.LOG = []

    def init_env(self, grid_size: int=20, num_colors: int=4) -> None:
        # Start middle bottom of the env looking up
        agent_start_pos = (grid_size//2, grid_size-2) # tuple(np.random.randint(1, grid_size - 1, size=2))
        agent_start_dir = 3 # np.random.randint(0, 4)

        if self.env_type == 'MultiGoalsEnv':
            self.env = MultiGoalsEnv(render_mode = self.render_mode,
                                    agent_goal=self.goal_color,
                                    size=grid_size,
                                    agent_start_pos=agent_start_pos,
                                    agent_start_dir=agent_start_dir,
                                    agent_view_size=self.receptive_field,
                                    num_colors=num_colors,
                                    max_steps=self.max_steps)
            
        elif self.env_type == 'MultiRoomsGoalsEnv':
            self.env = MultiRoomsGoalsEnv(render_mode = self.render_mode,
                        agent_goal=self.goal_color,
                        size=grid_size,
                        agent_start_pos=agent_start_pos,
                        agent_start_dir=agent_start_dir,
                        agent_view_size=self.receptive_field,
                        num_colors=num_colors,
                        max_steps=self.max_steps)
        else:
            raise ValueError('Unknown environment type')
        
        self.max_steps = self.env.max_steps

    def reset(self) -> None:
        # Reset env
        self.env.step_count = 0
        self.env.reset_agent_pos()
        self.env.reset_grid()

        # Reset what the learner knows about the env
        self.reached_subgoal = False

        self.actions = SimpleQueue()
        self.transitions = SimpleQueue()

        self.reward = 0
        self.terminated = False
        self.going_to_subgoal = False
        self.going_to_goal = False

        self.obstacle_grid = np.ones((self.env.height, self.env.width))
        self.shortest_path_subgoal = None
        self.shortest_path_goal = None

        # Init with uniform beliefs (no prior)
        self.beliefs = 1 / 4 * np.ones((self.grid_size, self.grid_size, 4))

    def change_receptive_field(self, new_receptive_field: int) -> None:
        self.env.agent_view_size = new_receptive_field
        self.receptive_field = new_receptive_field
        if new_receptive_field >= self.env.height:
            self.env.see_through_walls = True
        else:
            self.env.see_through_walls = False

    def reset(self) -> None:
        self.env.reset_grid()
        self.env.agent_view_size = self.receptive_field
        if self.receptive_field >= self.env.height:
            self.env.see_through_walls = True
        else:
            self.env.see_through_walls = False

        # Init with uniform beliefs (no prior)
        self.beliefs = 1 / 4 * np.ones((self.env.height, self.env.height, 4))

        # Reset
        self.reached_subgoal = False

        self.actions = SimpleQueue()
        self.transitions = SimpleQueue()

        self.reward = 0
        self.terminated = False
        self.going_to_subgoal = False
        self.going_to_goal = False

        self.obstacle_grid = np.ones((self.env.height, self.env.width))
        self.shortest_path_subgoal = None
        self.shortest_path_goal = None
    
    def play(self, size: int=None) -> list:
        
        if size is None:
            size = self.max_steps

        actions = []
        for _ in range(size):

            self.LOG.append(f't={self.env.step_count}')
            
            a = self.policy()

            self.LOG.append(f'action {a}')

            actions.append(a)
            
            action = Actions(a)
            obs, reward, terminated, _, _ = self.env.step(action)
            self.update_beliefs(obs['image'])
            self.reward = reward
            self.terminated = terminated

            # Save for rendering
            if self.save_render:
                self.render_frames.append(self.env.render())
                beliefs_image = Shannon_entropy(self.beliefs, axis=2).T
                self.render_beliefs.append(np.repeat(np.repeat(beliefs_image, 20, axis=0), 20, axis=1))

            if terminated:
                break


        if self.save_render:

            # Save frames
            gif_file = f"./outputs_rendering/output.gif"
            pil_images = [Image.fromarray(image.astype('uint8'), 'RGB') for image in self.render_frames]
            pil_images[0].save(gif_file, save_all=True, append_images=pil_images[1:], duration=100, loop=0)

            # Save associeted beliefs entropy
            gif_file = f"./outputs_rendering/output_belief.gif"
            pil_images = [Image.fromarray(image / (Shannon_entropy( 1 / 4 * np.ones(4)) + 0.5) * 255) for image in self.render_beliefs]
            pil_images[0].save(gif_file, save_all=True, append_images=pil_images[1:], duration=100, loop=0)

        return actions


    def update_beliefs(self, obs: np.ndarray) -> None:
        f_vec = self.env.dir_vec
        r_vec = self.env.right_vec
        top_left = (
            self.env.agent_pos
            + f_vec * (self.receptive_field - 1)
            - r_vec * (self.receptive_field // 2)
        )
        # Compute which cells are visible to the agent
        _, vis_mask = self.env.gen_obs_grid()

        # For each cell in the visibility mask
        for vis_j in range(0, self.receptive_field):
            for vis_i in range(0, self.receptive_field):
                
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                if abs_i < 0 or abs_i >= self.env.width:
                    continue
                if abs_j < 0 or abs_j >= self.env.height:
                    continue

                # Goal (door)
                if (obs[vis_i, vis_j, :2] == np.array([4, self.goal_color])).all() and (self.env.agent_pos != (abs_i, abs_j)):
                    self.beliefs[abs_i, abs_j, :] = np.array([0, 0, 1, 0])
                # Subgoal (key)
                elif (obs[vis_i, vis_j, :2] == np.array([5, self.goal_color])).all() and (self.env.agent_pos != (abs_i, abs_j)):
                    self.beliefs[abs_i, abs_j, :] = np.array([0, 0, 0, 1])
                # Obstacle
                elif obs[vis_i, vis_j, 0] in [2, 4, 5] and (self.env.agent_pos != (abs_i, abs_j)):
                    self.beliefs[abs_i, abs_j, :] = np.array([0, 1, 0, 0])
                # Nothing
                else:
                    self.beliefs[abs_i, abs_j, :] = np.array([1, 0, 0, 0])
    
    def compute_exploration_score(self, dir: int, pos: tuple) -> float:
        f_vec = DIR_TO_VEC[dir]
        dir_vec = DIR_TO_VEC[dir]
        dx, dy = dir_vec
        r_vec =  np.array((-dy, dx))
        top_left = (
            pos
            + f_vec * (self.receptive_field - 1)
            - r_vec * (self.receptive_field // 2)
        )
        exploration_score = 0

        # Compute what can be seen by the learner
        topX, topY, _, _ = get_view(pos, dir, self.receptive_field)
        grid = self.env.grid.slice(topX, topY, self.receptive_field, self.receptive_field)
        for _ in range(dir + 1):
            grid = grid.rotate_left()
        vis_mask = grid.process_vis(agent_pos=(self.receptive_field // 2, self.receptive_field - 1))

        # For each cell in the visibility mask
        for vis_j in range(0, self.receptive_field):
            for vis_i in range(0, self.receptive_field):

                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                if abs_i < 0 or abs_i >= self.env.width:
                    continue
                if abs_j < 0 or abs_j >= self.env.height:
                    continue
                exploration_score += Shannon_entropy(self.beliefs[abs_i, abs_j, :])

        return exploration_score
        
    def exploration_goal(self) -> int:
        # Unexplored locations
        unexplored_pos = np.where(Shannon_entropy(self.beliefs, axis=2) != 0)
        # At least one unexplored position
        if len(unexplored_pos[0]) > 0:
            # Manhattan distance to the unexplored locations
            dist = np.array([Manhattan_dist(self.env.agent_pos, pos) for pos in zip(unexplored_pos[0], unexplored_pos[1])])
            # Closest unexplored locations
            argmin_set = np.where(np.isclose(dist, np.min(dist)))[0]
            dest_idx = np.random.choice(argmin_set)
            dest_pos = (unexplored_pos[0][dest_idx], unexplored_pos[1][dest_idx])

            # Obstacle grid
            grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
            grid[dest_pos[0], dest_pos[1]] = 0
            
            # If the intermediate exploratory goal is not reacheable change exploratory goal
            while dist[dest_idx] < 10e10:
                path = A_star_algorithm(self.env.agent_pos, dest_pos, grid)
                if path is None:
                    dist[dest_idx] = 10e10
                    argmin_set = np.where(np.isclose(dist, np.min(dist)))[0]
                    dest_idx = np.random.choice(argmin_set)
                    dest_pos = (unexplored_pos[0][dest_idx], unexplored_pos[1][dest_idx])
                    grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
                    grid[dest_pos[0], dest_pos[1]] = 0
                else:
                    # Add transitions to go to exploratory goal
                    self.LOG.append('Exploratory goal')
                    mapped_actions = map_actions(self.env.agent_pos, path[0][1], self.env.agent_dir)
                    self.actions.put(mapped_actions[0])
                    # self.transitions.put(path[0])
                    return self.policy()
                
        return self.active_exploration_policy(forced=True)
    
    def active_exploration_policy(self, forced: bool=False) -> int:

        self.LOG.append(f'Exploration forced={forced}')

        # Action that maximizes the exploration
        scores = np.zeros(3)
        # Turn left
        scores[0] = self.compute_exploration_score(dir=(self.env.agent_dir - 1) % 4, pos=self.env.agent_pos)
        # Turn right
        scores[1] = self.compute_exploration_score(dir=(self.env.agent_dir + 1) % 4, pos=self.env.agent_pos)
        # Move forward
        next_pos = self.env.agent_pos + DIR_TO_VEC[self.env.agent_dir]
        if np.all(self.beliefs[next_pos[0], next_pos[1], :] == np.array([0, 1, 0, 0])) or \
            np.all(self.beliefs[next_pos[0], next_pos[1], :] == np.array([0, 0, 1, 0])): # Obstacle in front
            scores[2] = -1.
        else:
            scores[2] = self.compute_exploration_score(dir=self.env.agent_dir, pos=next_pos)

        argmax_set = np.where(np.isclose(scores, np.max(scores)))[0]
        
        self.LOG.append(f'scores {scores}')

        # If actions better than the others
        if len(argmax_set) < 3 or forced:
            return np.random.choice(argmax_set)
        
        # If all the actions are equal --> select closest exploration goal
        else:
            return self.exploration_goal()

    def add_actions(self, pos_dest: tuple) -> None:
        # Mapping position transition --> actions
        actions = map_actions(self.env.agent_pos, pos_dest, self.env.agent_dir)
        for a in actions:
            self.actions.put(a)

    def obj_in_front(self, obj_idx: int) -> bool:

        dx, dy = 0, 0
        if self.env.agent_dir == 0:
            dx = 1
        elif self.env.agent_dir == 2:
            dx = -1
        elif self.env.agent_dir == 3:
            dy = -1
        elif self.env.agent_dir == 1:
            dy = 1

        agent_pos = self.env.agent_pos
        return self.beliefs[agent_pos[0] + dx, agent_pos[1] + dy, obj_idx] == 1

    def empty_queues(self):
        while not self.actions.empty():
            self.actions.get()
        while not self.transitions.empty():
            self.transitions.get()
        return self.policy()
        
    def policy(self):
        self.LOG.append('Enter policy')

        if self.env.step_count == 0:
            self.LOG.append('First')
            return 4 # unused (to get first observation)

        # Subgoal (key) in front of the agent
        if self.obj_in_front(obj_idx=3):
            self.LOG.append('Key in front')
            # Set variables
            self.reached_subgoal = True
            self.going_to_subgoal = False
            # Subgoal (key) reached --> empty queues
            while not self.transitions.empty():
                _ = self.transitions.get()
            while not self.actions.empty():
                _ = self.actions.get()
            # Pickup the subgoal (key)
            self.LOG.append('RETURN 3')
            return 3
        
        # Goal (door) in front of the agent
        if self.obj_in_front(obj_idx=2) and self.reached_subgoal:
            self.LOG.append('Door in front')
            # Set variables
            self.going_to_goal = False
            # Goal (door) reached --> empty queues
            while not self.transitions.empty():
                _ = self.transitions.get()
            while not self.actions.empty():
                _ = self.actions.get()
            # Open the goal (door)
            self.LOG.append('RETURN 5')
            return 5
        
        # Action to be played
        if not self.actions.empty():
            self.LOG.append('Action to be done')
            action = self.actions.get()
            return action
        
        # Position to be reached
        if not self.transitions.empty():
            self.LOG.append('Position to be reacher')
            pos_init, pos_dest = self.transitions.get()
            
            # Sanity check
            assert(pos_init == self.env.agent_pos)

            # If not an obstacle --> add action to reach pos_dest
            if not ((np.all(self.beliefs[pos_dest[0], pos_dest[1], :] == np.array([0, 1, 0, 0]))) \
                    or (np.all(self.beliefs[pos_dest[0], pos_dest[1], :] == np.array([0, 0, 1, 0])) and not self.reached_subgoal)):
                self.add_actions(pos_dest)

            return self.policy()

        # If know where is the subgoal (key) & not already have subgoal (key) --> go to the subgoal (key)
        if (self.beliefs[:, :, 3] == 1).any() and not self.reached_subgoal:
            self.LOG.append('Know where is the key')
            subgoal_pos = np.where(self.beliefs[:, :, 3] == 1)
            # Obstacle grid
            grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
            # Check if new info
            compute_shortest_path = False
            if np.any(grid != self.obstacle_grid):
                self.obstacle_grid = grid.copy()
                compute_shortest_path = True
            grid[subgoal_pos[0], subgoal_pos[1]] = 0

            # First time compute path to subgoal
            if self.shortest_path_subgoal is None:
                self.shortest_path_subgoal = SimpleQueue()
                compute_shortest_path = True
                
            # If new info --> recompute shortest path
            if compute_shortest_path:
                path = A_star_algorithm(self.env.agent_pos, subgoal_pos, grid)
                if path is not None:
                    self.LOG.append('Recompute shortest path to subgoal')
                    # Empty pevious path
                    while not self.shortest_path_subgoal.empty():
                        _ = self.shortest_path_subgoal.get()
                    # Fill with new path
                    for transition in path:
                        self.shortest_path_subgoal.put(transition)

            if not self.shortest_path_subgoal.empty():
                self.LOG.append('Go to subgoal')
                # Add transition to go to subgoal (key)
                self.transitions.put(self.shortest_path_subgoal.get())
                # Set variable
                self.going_to_subgoal = True
                # Return action
                return self.policy()

        # If know where is the goal (door) & has subgoal (key) --> go to the goal (door)
        elif (self.beliefs[:, :, 2] == 1).any() and self.reached_subgoal:
            self.LOG.append('Know where is the door')
            goal_pos = np.where(self.beliefs[:, :, 2] == 1)
            # Obstacle grid
            grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
            # Check if new info
            compute_shortest_path = False
            if np.any(grid != self.obstacle_grid):
                self.LOG.append('New info (for goal)')
                self.obstacle_grid = grid.copy()
                compute_shortest_path = True
            grid[goal_pos[0], goal_pos[1]] = 0

            # First time compute path to the goal
            if self.shortest_path_goal is None:
                self.shortest_path_goal = SimpleQueue()
                compute_shortest_path = True

            # If new info --> recompute shortest path
            if compute_shortest_path:                
                path = A_star_algorithm(self.env.agent_pos, goal_pos, grid)
                if path is not None:
                    self.LOG.append('Recompute shortest path to goal')
                    # Empty pevious path
                    while not self.shortest_path_goal.empty():
                        _ = self.shortest_path_goal.get()
                    # Fill with new path
                    for transition in path:
                        self.shortest_path_goal.put(transition)

            if not self.shortest_path_goal.empty():
                self.LOG.append('Go to goal')
                self.transitions.put(self.shortest_path_goal.get())
                # Set variable
                self.going_to_goal = True
                # Return action
                return self.policy()
        
        # Nothing to do
        self.LOG.append('Active exploration')
        # Action that maximizes the exploration
        return self.active_exploration_policy(forced=True)
    
    def observe(self, traj: list, render_mode: str | None="rgb_array") -> None:
        if len(traj) > 0:
            # Add first unused action to get the first observation
            demo = [4] + traj
        else:
            demo = traj

        # Assert knows nothing on env
        assert((self.env.agent_pos == self.env.agent_start_pos) & (self.env.agent_dir == self.env.agent_start_dir))
        assert(np.all(self.beliefs == 1 / 4))

        # For rendering
        self.render_frames_observation = []
        self.render_beliefs_observation = []
        self.pos = []

        vis_mask = np.ones((self.grid_size, self.grid_size))
        # Follow traj and update beliefs
        for a in demo:
            # For rendering
            if render_mode == "rgb_array":
                self.render_frames_observation.append(self.env.render())
                beliefs_image = Shannon_entropy(self.beliefs, axis=2) / (Shannon_entropy( 1 / 4 * np.ones(4)) + 0.2)
                self.render_beliefs_observation.append(beliefs_image.T)
            else:
                grid = self.env.grid.slice(0, 0, self.grid_size, self.grid_size)
                image = grid.encode(vis_mask)
                image[self.env.agent_pos[0], self.env.agent_pos[1]] = np.array([OBJECT_TO_IDX['agent'], 0 , 0])
                self.render_frames_observation.append(image)
            self.pos.append(self.env.agent_pos)

            action = Actions(a)
            obs, _, _, _, _ = self.env.step(action)
            self.update_beliefs(obs['image'])

        # Reset learner init pos
        self.env.reset_grid()
##
# Learner that plans multiple position transitions (sequence planning)
##

# class BayesianLearnerPlanning:

#     def __init__(self,
#                  goal_color: int,
#                  receptive_field: int,
#                  grid_size: int=20,
#                  num_colors: int=4,
#                  save_render: bool=False,
#                  max_steps: int=200
#                  ) -> None:
        
#         self.goal_color = goal_color + 1
#         self.receptive_field = receptive_field
#         self.max_steps = max_steps

#         # Init with uniform beliefs (no prior)
#         self.beliefs = 1 / 4 * np.ones((grid_size, grid_size, 4))

#         # Generate feasable environment
#         feasible = False
#         while not feasible:

#             # Generate env
#             self.init_env(grid_size=grid_size,
#                           num_colors=num_colors)
#             self.env.reset()

#             # Check feasability
#             grid = np.ones((self.env.height, self.env.width))
#             for i in range(self.env.height):
#                 for j in range(self.env.width):
#                     grid[i, j] = self.env.grid.get(i,j) is not None

#             subgoal_pos = self.env.obj_idx[2 * self.goal_color]
#             grid[subgoal_pos[0], subgoal_pos[1]] = 0
#             path_subgoal = A_star_algorithm(start=self.env.agent_pos, goal=subgoal_pos, grid=grid)

#             goal_pos = self.env.obj_idx[2 * self.goal_color - 1]
#             grid[goal_pos[0], goal_pos[1]] = 0
#             path_goal = A_star_algorithm(start=self.env.agent_pos, goal=goal_pos, grid=grid)

#             feasible = path_goal is not None and path_subgoal is not None
        
#         self.reached_subgoal = False

#         self.actions = SimpleQueue()
#         self.transitions = SimpleQueue()

#         self.reward = 0
#         self.terminated = False
#         self.going_to_subgoal = False
#         self.going_to_goal = False

#         # For rendering
#         self.save_render = save_render
#         self.render_frames = []
#         self.render_beliefs = []

#     def init_env(self, grid_size: int=20, num_colors: int=4) -> None:
#         agent_start_pos = (grid_size//2, grid_size-2) # tuple(np.random.randint(1, grid_size - 1, size=2))
#         agent_start_dir = 3 # np.random.randint(0, 4)

#         self.env = MultiGoalsEnv(render_mode = "rgb_array",
#                                  agent_goal=self.goal_color,
#                                  size=grid_size,
#                                  agent_start_pos=agent_start_pos,
#                                  agent_start_dir=agent_start_dir,
#                                  agent_view_size=self.receptive_field,
#                                  num_colors=num_colors,
#                                  max_steps=self.max_steps)
    
#     def play(self, size: int=None) -> list:
        
#         if size is None:
#             size = self.max_steps

#         actions = []
#         for _ in range(size):
#             a = self.policy()
#             actions.append(a)
            
#             action = Actions(a)
#             obs, reward, terminated, _, _ = self.env.step(action)
#             self.update_beliefs(obs['image'])
#             self.reward = reward
#             self.terminated = terminated

#             # Save for rendering
#             self.render_frames.append(self.env.render())
#             beliefs_image = Shannon_entropy(self.beliefs, axis=2).T
#             self.render_beliefs.append(np.repeat(np.repeat(beliefs_image, 20, axis=0), 20, axis=1))

#             if terminated:
#                 break

#         if self.save_render:

#             # Save frames
#             gif_file = f"./outputs_rendering/output.gif"
#             pil_images = [Image.fromarray(image.astype('uint8'), 'RGB') for image in self.render_frames]
#             pil_images[0].save(gif_file, save_all=True, append_images=pil_images[1:], duration=100, loop=0)

#             # Save associeted beliefs entropy
#             gif_file = f"./outputs_rendering/output_belief.gif"
#             pil_images = [Image.fromarray(image / (Shannon_entropy( 1 / 4 * np.ones(4)) + 0.5) * 255) for image in self.render_beliefs]
#             pil_images[0].save(gif_file, save_all=True, append_images=pil_images[1:], duration=100, loop=0)

#         return actions


#     def update_beliefs(self, obs: np.ndarray) -> None:
#         f_vec = self.env.dir_vec
#         r_vec = self.env.right_vec
#         top_left = (
#             self.env.agent_pos
#             + f_vec * (self.receptive_field - 1)
#             - r_vec * (self.receptive_field // 2)
#         )

#         # Compute which cells are visible to the agent
#         _, vis_mask = self.env.gen_obs_grid()

#         # For each cell in the visibility mask
#         for vis_j in range(0, self.receptive_field):
#             for vis_i in range(0, self.receptive_field):
                
#                 if not vis_mask[vis_i, vis_j]:
#                     continue

#                 # Compute the world coordinates of this cell
#                 abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
#                 if abs_i < 0 or abs_i >= self.env.width:
#                     continue
#                 if abs_j < 0 or abs_j >= self.env.height:
#                     continue

#                 # Goal (door)
#                 if (obs[vis_i, vis_j, :2] == np.array([4, self.goal_color])).all() and (self.env.agent_pos != (abs_i, abs_j)):
#                     self.beliefs[abs_i, abs_j, :] = np.array([0, 0, 1, 0])
#                 # Subgoal (key)
#                 elif (obs[vis_i, vis_j, :2] == np.array([5, self.goal_color])).all() and (self.env.agent_pos != (abs_i, abs_j)):
#                     self.beliefs[abs_i, abs_j, :] = np.array([0, 0, 0, 1])
#                 # Obstacle
#                 elif obs[vis_i, vis_j, 0] in [2, 4, 5] and (self.env.agent_pos != (abs_i, abs_j)):
#                     self.beliefs[abs_i, abs_j, :] = np.array([0, 1, 0, 0])
#                 # Nothing
#                 else:
#                     self.beliefs[abs_i, abs_j, :] = np.array([1, 0, 0, 0])
    
#     def compute_exploration_score(self, dir: int, pos: tuple) -> float:
#         f_vec = DIR_TO_VEC[dir]
#         dir_vec = DIR_TO_VEC[dir]
#         dx, dy = dir_vec
#         r_vec =  np.array((-dy, dx))

#         top_left = (
#             pos
#             + f_vec * (self.receptive_field - 1)
#             - r_vec * (self.receptive_field // 2)
#         )

#         exploration_score = 0

#         # For each cell in the visibility mask
#         for vis_j in range(0, self.receptive_field):
#             for vis_i in range(0, self.receptive_field):

#                 # Compute the world coordinates of this cell
#                 abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
#                 if abs_i < 0 or abs_i >= self.env.width:
#                     continue
#                 if abs_j < 0 or abs_j >= self.env.height:
#                     continue

#                 exploration_score += Shannon_entropy(self.beliefs[abs_i, abs_j, :])
            
#         return exploration_score
    
#     def best_exploration_action(self, forced: bool=False) -> int:

#         # Action that maximizes the exploration
#         scores = np.zeros(3)
#         # Turn left
#         scores[0] = self.compute_exploration_score(dir=(self.env.agent_dir - 1) % 4, pos=self.env.agent_pos)
#         # Turn right
#         scores[1] = self.compute_exploration_score(dir=(self.env.agent_dir + 1) % 4, pos=self.env.agent_pos)
#         # Move forward
#         next_pos = self.env.agent_pos + DIR_TO_VEC[self.env.agent_dir]
#         if np.all(self.beliefs[next_pos[0], next_pos[1], :] == np.array([0, 1, 0, 0])) or \
#             np.all(self.beliefs[next_pos[0], next_pos[1], :] == np.array([0, 0, 1, 0])): # Obstacle in front
#             scores[2] = 0.
#         else:
#             scores[2] = self.compute_exploration_score(dir=self.env.agent_dir, pos=next_pos)

#         argmax_set = np.where(np.isclose(scores, np.max(scores)))[0]

#         # If actions better than the others
#         if len(argmax_set) < 3 or forced:
#             best_action =  np.random.choice(argmax_set)
#             return best_action
        
#         # If all the actions are equal
#         else:

#             # Unexplored locations
#             unexplored_pos = np.where(Shannon_entropy(self.beliefs, axis=2) != 0)
#             # At least one unexplored position
#             if len(unexplored_pos[0]) > 0:
#                 # Manhattan distance to the unexplored locations
#                 dist = np.array([Manhattan_dist(self.env.agent_pos, pos) for pos in zip(unexplored_pos[0], unexplored_pos[1])])
#                 # Closest unexplored locations
#                 argmin_set = np.where(np.isclose(dist, np.min(dist)))[0]
#                 dest_idx = np.random.choice(argmin_set)
#                 dest_pos = (unexplored_pos[0][dest_idx], unexplored_pos[1][dest_idx])

#                 # Obstacle grid
#                 grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
#                 grid[dest_pos[0], dest_pos[1]] = 0
                
#                 # If the intermediate exploratory goal is not reacheable change exploratory goal
#                 while dist[dest_idx] < 10e5:
#                     path = A_star_algorithm(self.env.agent_pos, dest_pos, grid)
#                     if path is None:
#                         dist[dest_idx] = 10e5
#                         argmin_set = np.where(np.isclose(dist, np.min(dist)))[0]
#                         dest_idx = np.random.choice(argmin_set)
#                         dest_pos = (unexplored_pos[0][dest_idx], unexplored_pos[1][dest_idx])
#                         grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
#                         grid[dest_pos[0], dest_pos[1]] = 0
#                     else:
#                         # Add transitions to go to exploratory goal
#                         for transition in path:
#                             self.transitions.put(transition)

#                         return self.policy()

#             return self.best_exploration_action(forced=True)
            

#     def add_actions(self, pos_dest: tuple) -> None:
#         # Mapping position transition --> actions
#         actions = map_actions(self.env.agent_pos, pos_dest, self.env.agent_dir)
#         for a in actions:
#             self.actions.append(a)
    
#     def obj_in_front(self, obj_idx: int) -> bool:

#         dx, dy = 0, 0
#         if self.env.agent_dir == 0:
#             dx = 1
#         elif self.env.agent_dir == 2:
#             dx = -1
#         elif self.env.agent_dir == 3:
#             dy = -1
#         elif self.env.agent_dir == 1:
#             dy = 1

#         agent_pos = self.env.agent_pos
#         return self.beliefs[agent_pos[0] + dx, agent_pos[1] + dy, obj_idx] == 1
        
#     def policy(self):
            
#         if self.env.step_count == 0:
#             return 4 # unused (to get first observation)

#         # Subgoal (key) in front of the agent
#         if self.obj_in_front(obj_idx=3):

#             self.reached_subgoal = True
#             self.going_to_subgoal = False
#             # Subgoal (key) reached --> empty queues
#             while not self.transitions.empty():
#                 _ = self.transitions.get()
#             while not self.actions.empty():
#                 _ = self.actions.get()
#             # Pickup the subgoal (key)
#             return 3
        
#         # Goal (door) in front of the agent
#         if self.obj_in_front(obj_idx=2) and self.reached_subgoal:

#             self.going_to_goal = False
#             # Goal (door) reached --> empty queues
#             while not self.transitions.empty():
#                 _ = self.transitions.get()
#             while not self.actions.empty():
#                 _ = self.actions.get()
#             # Open the goal (door)
#             return 5

#         # If know where is the subgoal (key) & not already have subgoal (key) & not already going to the subgoal (key) --> go to the subgoal (key)
#         if (self.beliefs[:, :, 3] == 1).any() and not self.reached_subgoal and not self.going_to_subgoal:

#             subgoal_pos = np.where(self.beliefs[:, :, 3] == 1)
#             # Obstacle grid
#             grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
#             grid[subgoal_pos[0], subgoal_pos[1]] = 0
            
#             path = A_star_algorithm(self.env.agent_pos, subgoal_pos, grid)

#             if path is not None:
#                 # Empty queues
#                 while not self.transitions.empty():
#                     _ = self.transitions.get()
#                 while not self.actions.empty():
#                     _ = self.actions.get()
#                 # Add transitions to go to key
#                 for transition in path:
#                     self.transitions.put(transition)
#                 # Set variable
#                 self.going_to_subgoal = True
#                 # Return action
#                 return self.policy()

#         # If know where is the goal (door) & has subgoal (key) & not already going to the goal (door) --> go to the goal (door)
#         elif (self.beliefs[:, :, 2] == 1).any() and self.reached_subgoal and not self.going_to_goal:

#             goal_pos = np.where(self.beliefs[:, :, 2] == 1)
#             # Obstacle grid
#             grid = np.ones((self.env.width, self.env.height)) - np.all(self.beliefs == np.array([1., 0., 0., 0.]).reshape(1, 1, -1), axis=2)
#             grid[goal_pos[0], goal_pos[1]] = 0

#             path = A_star_algorithm(self.env.agent_pos, goal_pos, grid)

#             if path is not None:
#                 # Empty queues
#                 while not self.transitions.empty():
#                     _ = self.transitions.get()
#                 while not self.actions.empty():
#                     _ = self.actions.get()
#                 # Add transitions to go to goal (door)
#                 for transition in path:
#                     self.transitions.put(transition)
#                 # Set variable
#                 self.going_to_goal = True
#                 # Return action
#                 return self.policy()
        
#         # Action to be played
#         if not self.actions.empty():
#             action = self.actions.get()
#             return action
        
#         # Position to be reached
#         if not self.transitions.empty():
#             pos_init, pos_dest = self.transitions.get()
            
#             # Sanity check
#             assert(pos_init == self.env.agent_pos)

#             # If not an obstacle --> add action to reach pos_dest
#             if not ((np.all(self.beliefs[pos_dest[0], pos_dest[1], :] == np.array([0, 1, 0, 0]))) \
#                     or (np.all(self.beliefs[pos_dest[0], pos_dest[1], :] == np.array([0, 0, 1, 0])) and not self.reached_subgoal)):
#                 self.add_actions(pos_dest)

#             return self.policy()
        
#         # Nothing to do 
#         else:
#             # Action that maximizes the exploration
#             return self.best_exploration_action()