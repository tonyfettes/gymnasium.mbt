.PHONY: default target/wasm/release/build/gen/gen.wat

default: trainer

target/wasm/release/build/gen/gen.wat:
	moon build --target wasm --output-wat

target/wasm/release/build/gen/gen.wasm: wit/world.wit target/wasm/release/build/gen/gen.wat
	wasm-tools component embed $^ -o $@ --encoding utf16

target/wasm/release/build/gen/gen.component.wasm: target/wasm/release/build/gen/gen.wasm
	wasm-tools component new $< -o $@

trainer: target/wasm/release/build/gen/gen.component.wasm
	.venv/bin/python3.11 -m wasmtime.bindgen $< --out-dir $@
