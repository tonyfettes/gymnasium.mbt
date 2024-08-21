struct Q {
  action_space : Int
  random_state : @random.RandomState
  table : Array[Array[Double]]
  mut epsilon : Double
  epsilon_decay : Double
  learning_rate : Double
  discount_factor : Double
}

fn Q::new(
  ~observation_space : Int,
  ~action_space : Int,
  ~seed : Int,
  ~epsilon : Double,
  ~epsilon_decay : Double,
  ~learning_rate : Double,
  ~discount_factor : Double
) -> Q {
  let random_state = @random.init_state(~seed)
  let table = Array::makei(
    observation_space,
    fn(_i) { Array::make(action_space, 0.5) },
  )
  Q::{
    random_state,
    action_space,
    table,
    epsilon,
    epsilon_decay,
    learning_rate,
    discount_factor,
  }
}

fn random_sample(
  random_state : @random.RandomState,
  distribution : Array[Double]
) -> Int {
  let mut sum = 0.0
  for p in distribution {
    sum = sum + p + 1.0
  }
  let mut value = @random.gen_double(random_state) / 2.0 * sum
  for i = 0; i < distribution.length(); i = i + 1 {
    value = value - distribution[i]
    if value < 0 {
      return i
    }
  }
  return distribution.length() - 1
}

fn argmax[T : Compare](array : Array[T]) -> Int {
  let mut max_value = array[0]
  let mut max_index = 0
  for i = 1; i < array.length(); i = i + 1 {
    if array[i] > max_value {
      max_index = i
      max_value = array[i]
    }
  }
  max_index
}

fn max[T : Compare](array : Array[T]) -> T {
  let mut max_value = array[0]
  for i = 1; i < array.length(); i = i + 1 {
    if array[i] > max_value {
      max_value = array[i]
    }
  }
  max_value
}

fn Q::sample_action(self : Q, state : Int) -> UInt {
  {
    let table = self.table
    @python.print("\{table}".to_bytes())
  }
  if @random.gen_double(self.random_state) / 2.0 < self.epsilon {
    @python.print("explore".to_bytes())
    return (@random.gen_int(self.random_state) % self.action_space).to_uint()
  }
  @python.print("exploit".to_bytes())
  random_sample(self.random_state, self.table[state]).to_uint()
}

fn Q::update_reward(
  self : Q,
  state : Int,
  action : UInt,
  result : Int,
  reward : Double
) -> Unit {
  let action = action.to_int()
  let old_value = self.table[state][action]
  self.table[state][action] = self.table[state][action] +
    self.learning_rate *
    (
      reward +
      self.discount_factor * max(self.table[result]) -
      self.table[state][action]
    )
  let new_value = self.table[state][action]
  @python.print(
    "updating table[\{state}][\{action}] from \{old_value} to \{new_value}".to_bytes(),
  )
  self.epsilon = self.epsilon * self.epsilon_decay
  let epsilon = self.epsilon
  @python.print("epsilon: \{epsilon}".to_bytes())
}

pub fn run() -> Unit {
  let env = @gymnasium.frozen_lake_make("human".to_bytes(), false)
  let agent = Q::new(
    observation_space=env.observation_space.n.to_int(),
    action_space=env.action_space.n.to_int(),
    seed=42,
    epsilon=1.0,
    epsilon_decay=0.99,
    learning_rate=1.0,
    discount_factor=0.99,
  )
  for episode = 0; episode < 100; episode = episode + 1 {
    let (init_observation, _) = @gymnasium.frozen_lake_reset(env, None)
    let mut last_observation = init_observation
    for tick = 0; tick < 1024; tick = tick + 1 {
      let action = agent.sample_action(last_observation)
      @python.print("action: \{action}".to_bytes())
      let (next_observation, reward, terminated, _) = @gymnasium.frozen_lake_step(
        env, action,
      )
      let reward = if reward == 0.0 { -0.01 } else { reward }
      @python.print(
        "observation: \{next_observation}, reward: \{reward}".to_bytes(),
      )
      agent.update_reward(last_observation, action, next_observation, reward)
      last_observation = next_observation
      if terminated {
        break
      }
    }
  }
}
