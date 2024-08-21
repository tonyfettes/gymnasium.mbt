WIT = wit/world.wit
WAT = target/wasm/release/build/gen/gen.wat

.PHONY: default $(WAT) clean run

default: main

ffi gen interface worlds &: $(WIT)
	wit-bindgen moonbit wit --out-dir . --derive-show --derive-eq --ignore-stub
	moon fmt

$(WAT):
	moon build --target wasm --output-wat

target/wasm/release/build/gen/gen.embedded.wasm: $(WIT) $(WAT)
	wasm-tools component embed $(WIT) $(WAT) -o $@ --encoding utf16

target/wasm/release/build/gen/gen.component.wasm: target/wasm/release/build/gen/gen.embedded.wasm
	wasm-tools component new $< -o $@

target/wasm/release/build/gen/gen.component.wat: target/wasm/release/build/gen/gen.component.wasm
	wasm-tools print $< > $@

main: target/wasm/release/build/gen/gen.component.wasm
	python3 -m wasmtime.bindgen $< --out-dir $@

clean:
	rm -rf ffi gen interface target

run: main main.py
	python3 main.py
