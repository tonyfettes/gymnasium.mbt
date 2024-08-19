default: target/gen.component.wasm

target/wasm/release/build/gen/gen.wasm:
	moon build --target wasm

target/gen.wasm: target/wasm/release/build/gen/gen.wasm
	wasm-tools component embed wit target/wasm/release/build/gen/gen.wasm -o target/gen.wasm --encoding utf16

target/gen.component.wasm: target/gen.wasm
	wasm-tools component new target/gen.wasm -o target/gen.component.wasm
