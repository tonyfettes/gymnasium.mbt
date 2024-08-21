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
      @python.print(
        "origin: \{origin} --(\{action})-> state: \{state}".to_bytes(),
      )
      continue trajectory, state
    }
    [], _ => ()
  }
}

fn Q::update_reward(self : Q, state : Int, reward : Double) -> Unit {
  debug("learning_rate", self.learning_rate)
  debug("epsilon", self.epsilon.epsilon)
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

pub fn run() -> Unit {
  let env = @gymnasium.frozen_lake_make("human".to_bytes(), false)
  let agent = Q::new(
    observation_space=env.observation_space.n.to_int(),
    action_space=env.action_space.n.to_int(),
    seed=42,
    epsilon=Epsilon::new(epsilon=1.0, bias=0.10, decay=0.999),
    learning_rate=1.0,
    discount_factor=0.99,
  )
  for episode = 0; episode < 100; episode = episode + 1 {
    let (observation, _) = @gymnasium.frozen_lake_reset(env, None)
    loop observation {
      observation => {
        let action = agent.sample_action(observation)
        let (observation, reward, terminated, _) = @gymnasium.frozen_lake_step(
          env, action,
        )
        agent.update_reward(observation, reward)
        if terminated {
          break
        }
        continue observation
      }
    }
  }
}
