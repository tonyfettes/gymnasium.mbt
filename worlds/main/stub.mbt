struct Q {
  random_state: @random.RandomState
  table: Array[Array[Double]]
}

fn Q::new(observation_space: Int, action_space: Int) -> Q {
  let table = Array::makei(observation_space, fn(_i) { Array::make(action_space, 0.0) })
  let random_state = @random.new()
  Q::{
    random_state,
    table
  }
}

fn Q::sample_action(self: Q) -> Int {
  let epsilon_value = @random.gen_double(self.random_state)
  0
}

pub fn run() -> Unit {
  let env = @gymnasium.frozen_lake_make("human".to_bytes(), false)
  for episode = 0; episode < 100; episode = episode + 1 {
    let (observation, info) = @gymnasium.frozen_lake_reset(env, None)
    @python.print("observation: \{observation}".to_bytes())
    for tick = 0; tick < 1024; tick = tick + 1 {
      let action = @gymnasium.discrete_sample(env.action_space)
      @python.print("action: \{action}".to_bytes())
      let (observation, reward, terminated, info) = @gymnasium.frozen_lake_step(
        env, action,
      )
      @python.print("observation: \{observation}, reward: \{reward}".to_bytes())
      if terminated {
        break
      }
    }
  }
}
