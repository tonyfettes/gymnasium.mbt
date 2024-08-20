from main import Root, RootImports, imports
import gymnasium as gym
from typing import Any, List, Tuple
import wasmtime
import numpy

from main.imports.gymnasium import Discrete


class HostGymnasium(imports.HostGymnasium):
    environments: list[gym.Env[Any, Any]]
    spaces: list[gym.Space]

    def __init__(self) -> None:
        super().__init__()
        self.environments = []
        self.spaces = []

    def frozen_lake_make(
        self, render_mode: bytes, is_slippery: bool
    ) -> imports.gymnasium.FrozenLake:
        env_id = len(self.environments)
        render_mode_str = render_mode.decode("utf-16")
        env: gym.Env[gym.spaces.Discrete, gym.spaces.Discrete] = gym.make(
            "FrozenLake-v1", render_mode=render_mode_str, is_slippery=is_slippery
        )
        self.environments.append(env)
        action_space_id = len(self.spaces)
        self.spaces.append(env.action_space)
        observation_space_id = len(self.spaces)
        self.spaces.append(env.observation_space)
        return imports.gymnasium.FrozenLake(
            id=env_id,
            action_space=imports.gymnasium.Discrete(id=action_space_id, n=env.action_space.n),  # type: ignore
            observation_space=imports.gymnasium.Discrete(id=observation_space_id, n=env.observation_space.n),  # type: ignore
        )

    def frozen_lake_reset(
        self, env: imports.gymnasium.FrozenLake, seed: int
    ) -> Tuple[int, imports.gymnasium.FrozenLakeInfo]:
        observation, info = self.environments[env.id].reset(seed=seed)
        return observation, imports.gymnasium.FrozenLakeInfo(prob=info["prob"])

    def frozen_lake_step(
        self, env: imports.gymnasium.FrozenLake, action: int
    ) -> Tuple[int, float, bool, imports.gymnasium.FrozenLakeInfo]:
        observation, reward, terminated, truncated, info = self.environments[
            env.id
        ].step(action)
        return (
            observation,
            float(reward),
            terminated or truncated,
            imports.gymnasium.FrozenLakeInfo(prob=info["prob"]),
        )

    def discrete_sample(self, discrete: Discrete) -> int:
        return self.spaces[discrete.id].sample().item()


class HostPython(imports.HostPython):
    def print(self, message: bytes) -> None:
        print(message.decode("utf-16"))


def main():
    store = wasmtime.Store()
    trainer = Root(store, RootImports(python=HostPython(), gymnasium=HostGymnasium()))
    trainer.run(store)


if __name__ == "__main__":
    main()
