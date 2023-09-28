from __future__ import annotations
from minigrid.manual_control import ManualControl
from environment import MultiGoalsEnv


def main():
    env = MultiGoalsEnv(render_mode="human", agent_goal=0, agent_view_size=3)

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()
