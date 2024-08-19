fn main {
  let random_state = @random.new()
  let space = @gymnasium.make("LunarLander-v3")
  let _ = space.reset(42)
  for i = 0; i < 1000; i = i + 1 {
    let action : FixedArray[Double] = [
      (@random.gen_int(random_state) % 4).to_double(),
    ]
    let (_, stopped) = space.step(action)
    if stopped {
      break
    }
  }
}
