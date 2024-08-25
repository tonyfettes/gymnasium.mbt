let rand : @random.Rand = @random.new()

enum EpsilonPolicy {
  Explore
  Exploit
}

struct Epsilon {
  mut value : Double
  bias : Double
  decay : Double
}

fn Epsilon::new(value : Double, ~bias : Double = 0, ~decay : Double) -> Epsilon {
  Epsilon::{ value, bias, decay }
}

fn Epsilon::sample(self : Epsilon) -> EpsilonPolicy {
  if rand.double() < self.value {
    EpsilonPolicy::Explore
  } else {
    EpsilonPolicy::Exploit
  }
}

fn Epsilon::update(self : Epsilon) -> Unit {
  self.value = (self.value - self.bias) * self.decay + self.bias
}

struct Q {
  action_space : Int
  table : Array[Array[Double]]
  epsilon : Epsilon
  learning_rate : Double
  discount_factor : Double
  trajectory : Array[(Int, UInt)]
}

fn Q::new(
  ~observation_space : Int,
  ~action_space : Int,
  ~epsilon : Epsilon,
  ~learning_rate : Double,
  ~discount_factor : Double
) -> Q {
  let table = Array::makei(
    observation_space,
    fn(_i) { Array::make(action_space, 0.5) },
  )
  let trajectory = []
  Q::{
    action_space,
    table,
    epsilon,
    learning_rate,
    discount_factor,
    trajectory,
  }
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

fn random_sample(array : Array[Double]) -> Int {
  let mut sum = 0.0
  for value in array {
    sum = sum + value
  }
  let mut random_value = rand.double() * sum
  for i, value in array {
    random_value = random_value - value
    if random_value < 0 {
      return i
    }
  }
  abort("Invalid random value")
}

fn Q::sample_action(self : Q, state : Int) -> UInt {
  let action = match self.epsilon.sample() {
    EpsilonPolicy::Explore => rand.int(limit=self.action_space).to_uint()
    EpsilonPolicy::Exploit => argmax(self.table[state]).to_uint()
  }
  self.trajectory.push((state, action))
  return action
}

fn print_table(table : Array[Array[Double]]) -> Unit {
  for i, row in table {
    if i == 0 {
      @python.print("[\{row}".to_bytes())
    } else if i == table.length() - 1 {
      @python.print(" \{row}]".to_bytes())
    } else {
      @python.print(" \{row},".to_bytes())
    }
  }
}

fn debug[T : Show](label : String, value : T) -> Unit {
  @python.print("\{label}: \{value}".to_bytes())
}

fn print_trajectory(self : Q, state : Int) -> Unit {
  @python.print("trajectory:".to_bytes())
  loop self.trajectory[:], state {
    [(origin, action), .. as trajectory], state => {
      @python.print("  \{origin} --(\{action})-> \{state}".to_bytes())
      continue trajectory, state
    }
    [], _ => ()
  }
}

fn Q::update_reward(self : Q, state : Int, reward : Double) -> Unit {
  debug("learning_rate", self.learning_rate)
  debug("epsilon", self.epsilon.value)
  debug("reward", reward)
  self.print_trajectory(state)
  loop self.trajectory[:], state {
    [(origin, action), .. as trajectory], state => {
      // bellman equantion
      self.table[origin][action.to_int()] += self.learning_rate *
        (
          reward +
          self.discount_factor * max(self.table[state]) -
          self.table[origin][action.to_int()]
        )
      continue trajectory, state
    }
    [], _ => ()
  }
  self.epsilon.update()
  let table = self.table
  print_table(table)
  self.trajectory.clear()
}

struct ForzenLake {
  map : Array[String]
  env : @gymnasium.FrozenLake
}

fn observation_space(self : ForzenLake) -> @gymnasium.Discrete {
  self.env.observation_space
}

fn action_space(self : ForzenLake) -> @gymnasium.Discrete {
  self.env.action_space
}

fn ForzenLake::new(
  ~is_slippery : Bool = false,
  ~render_mode : String = "human",
  ~map? : Array[String]
) -> ForzenLake {
  let desc = match map {
    None => None
    Some(map) => Some(map.map(fn(row) { row.to_bytes() }))
  }
  return ForzenLake::{
    env: @gymnasium.frozen_lake_make(render_mode.to_bytes(), is_slippery, desc),
    map: match map {
      None => ["SFFF", "FHFH", "FFFH", "HFFG"]
      Some(map) => map
    },
  }
}

fn ForzenLake::get_map(self : ForzenLake, state : Int) -> Char {
  let mut state = state
  for row in self.map {
    if state < row.length() {
      return row[state]
    }
    state -= row.length()
  } else {
    abort("Invalid state: \{state}")
  }
}

fn ForzenLake::reset(
  self : ForzenLake,
  seed : UInt?
) -> (Int, @gymnasium.FrozenLakeInfo) {
  @gymnasium.frozen_lake_reset(self.env, seed)
}

fn ForzenLake::step(
  self : ForzenLake,
  action : UInt
) -> (Int, Double, Bool, @gymnasium.FrozenLakeInfo) {
  let (observation, reward, terminated, info) = @gymnasium.frozen_lake_step(
    self.env,
    action,
  )
  let reward = match self.get_map(observation) {
    'H' => -1.0
    'F' => -0.01
    'S' => -0.01
    _ => reward
  }
  return (observation, reward, terminated, info)
}

pub fn power(base : Double, exp : Int) -> Double {
  let mut result = 1.0
  for _i = 0; _i < exp; _i = _i + 1 {
    result = result * base
  }
  result
}

pub fn train_frozen_lake() -> Unit {
  let map = ["SFFF", "FHFH", "FFFH", "HFFG"]
  let env = ForzenLake::new(render_mode="human", is_slippery=false, ~map)
  let agent = Q::new(
    observation_space=env.observation_space().n.to_int(),
    action_space=env.action_space().n.to_int(),
    epsilon=Epsilon::new(1.0, decay=1.0 - power(0.5, map.length())),
    learning_rate=1.0,
    discount_factor=0.95,
  )
  for episode = 0; episode < 100; episode = episode + 1 {
    let (observation, _) = env.reset(None)
    loop observation {
      observation => {
        let action = agent.sample_action(observation)
        let (observation, reward, terminated, _) = env.step(action)
        agent.update_reward(observation, reward)
        if terminated {
          break
        }
        continue observation
      }
    }
  }
}

struct DQN {
  layers : @torch.Sequential
  learning_rate : Double
  discount_factor : Double
  log_probs : Array[@torch.Value]
  rewards : Array[Double]
}

pub fn DQN::new(
  ~observation_space : Int,
  ~action_space : Int,
  ~learning_rate : Double,
  ~discount_factor : Double
) -> DQN {
  let layers = @torch.Sequential::new(
    [
      @torch.Linear::new(observation_space, 16),
      @torch.ReLU::new(),
      @torch.Linear::new(16, action_space),
      @torch.Softmax::new(),
    ],
  )
  DQN::{ layers, learning_rate, discount_factor, log_probs: [], rewards: [] }
}

pub fn DQN::sample_action(self : DQN, state : Array[Double]) -> UInt {
  let value = self.layers.forward(@torch.Tensor::val(state))
  let distrib = @torch.Categorical::new(value)
  let action = distrib.sample()
  let log_prob = distrib.log_prob(action)
  self.log_probs.push(log_prob)
  action.to_uint()
}

pub fn DQN::update_reward(
  self : DQN,
  _state : Array[Double],
  reward : Double
) -> Unit {
  self.rewards.push(reward)
}

pub fn DQN::back_propagate(self : DQN) -> Double {
  let mut cumulative_reward = 0.0
  let deltas = []
  self.rewards.rev_inplace()
  for reward in self.rewards {
    cumulative_reward = reward + self.discount_factor * cumulative_reward
    deltas.push(cumulative_reward)
  }
  deltas.rev_inplace()
  debug("log_probs", self.log_probs)
  debug("deltas", deltas)
  let mut loss = @torch.Value::val(0.0)
  for i = 0; i < deltas.length(); i = i + 1 {
    loss += - self.log_probs[i] * @torch.Value::val(deltas[i])
  }
  loss.backward(self.learning_rate)
  @python.print("\{self.layers}".to_bytes())
  self.log_probs.clear()
  self.rewards.clear()
  return loss.value
}

pub struct LunarLander {
  env : @gymnasium.LunarLander
}

pub fn LunarLander::new(~render_mode : String) -> LunarLander {
  LunarLander::{ env: @gymnasium.lunar_lander_make(render_mode.to_bytes()) }
}

pub fn LunarLander::observation_space(self : LunarLander) -> @gymnasium.Box {
  self.env.observation_space
}

pub fn LunarLander::action_space(self : LunarLander) -> @gymnasium.Discrete {
  self.env.action_space
}

pub fn LunarLander::reset(self : LunarLander, seed : UInt?) -> Array[Double] {
  Array::from_fixed_array(@gymnasium.lunar_lander_reset(self.env, seed))
}

pub fn LunarLander::step(
  self : LunarLander,
  action : UInt
) -> (Array[Double], Double, Bool) {
  let (state, reward, terminated) = self.env
    |> @gymnasium.lunar_lander_step(action)
  (Array::from_fixed_array(state), reward, terminated)
}

pub fn train_lunar_lander() -> Unit {
  let env = LunarLander::new(render_mode="human")
  let agent = DQN::new(
    observation_space=env.observation_space().shape[0].to_int(),
    action_space=env.action_space().n.to_int(),
    learning_rate=1.0e-4,
    discount_factor=0.95,
  )
  for episode = 0; episode < 10000; episode = episode + 1 {
    loop env.reset(None) {
      state => {
        let action = agent.sample_action(state)
        let (state, reward, terminated) = env.step(action)
        agent.update_reward(state, reward)
        if terminated {
          break
        }
        continue state
      }
    }
    let loss = agent.back_propagate()
    debug("loss", loss)
  }
}

pub fn run() -> Unit {
  train_frozen_lake |> ignore
  train_lunar_lander()
}
