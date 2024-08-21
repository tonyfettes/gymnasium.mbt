enum EpsilonPolicy {
  Explore
  Exploit
}

struct Epsilon {
  mut epsilon : Double
  bias : Double
  decay : Double
}

fn Epsilon::new(~epsilon : Double, ~bias : Double, ~decay : Double) -> Epsilon {
  Epsilon::{ epsilon, bias, decay }
}

fn Epsilon::sample(
  self : Epsilon,
  random_state : @random.RandomState
) -> EpsilonPolicy {
  if @random.gen_double(random_state) < self.epsilon {
    EpsilonPolicy::Explore
  } else {
    EpsilonPolicy::Exploit
  }
}

fn Epsilon::update(self : Epsilon) -> Unit {
  self.epsilon = (self.epsilon - self.bias) * self.decay + self.bias
}

struct Q {
  action_space : Int
  random_state : @random.RandomState
  table : Array[Array[Double]]
  epsilon : Epsilon
  learning_rate : Double
  discount_factor : Double
  trajectory : Array[(Int, UInt)]
}

fn Q::new(
  ~observation_space : Int,
  ~action_space : Int,
  ~seed : Int,
  ~epsilon : Epsilon,
  ~learning_rate : Double,
  ~discount_factor : Double
) -> Q {
  let random_state = @random.init_state(~seed)
  let table = Array::makei(
    observation_space,
    fn(_i) { Array::make(action_space, 0.0) },
  )
  let trajectory = []
  Q::{
    random_state,
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

fn Q::sample_action(self : Q, state : Int) -> UInt {
  let action = match self.epsilon.sample(self.random_state) {
    EpsilonPolicy::Explore =>
      (@random.gen_int(self.random_state) % self.action_space).to_uint()
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
  debug("epsilon", self.epsilon.epsilon)
  debug("reward", reward)
  self.print_trajectory(state)
  loop self.trajectory[:], state {
    [(origin, action), .. as trajectory], state => {
      self.table[origin][action.to_int()] = self.table[origin][action.to_int()] +
        self.learning_rate *
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

pub fn run() -> Unit {
  let env = ForzenLake::new(
    render_mode="human",
    is_slippery=false,
    map=["SFFF", "FHFH", "FFFH", "HFFG"],
  )
  let agent = Q::new(
    observation_space=env.observation_space().n.to_int(),
    action_space=env.action_space().n.to_int(),
    seed=43,
    epsilon=Epsilon::new(epsilon=1.0, bias=0.10, decay=0.999),
    learning_rate=1.0,
    discount_factor=0.99,
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
