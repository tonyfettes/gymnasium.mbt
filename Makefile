WIT = wit/world.wit
WASM = target/wasm/debug/build/gen/gen.wasm

.PHONY: default $(WASM) clean run

default: main

ffi gen interface worlds &: $(WIT)
	wit-bindgen moonbit wit --out-dir . --derive-show --derive-eq --ignore-stub
	moon fmt

$(WASM):
	moon build --target wasm -g

target/wasm/debug/build/gen/gen.embedded.wasm: $(WIT) $(WASM)
	wasm-tools component embed $(WIT) $(WASM) -o $@ --encoding utf16

target/wasm/debug/build/gen/gen.component.wasm: target/wasm/debug/build/gen/gen.embedded.wasm
	wasm-tools component new $< -o $@

target/wasm/debug/build/gen/gen.component.wat: target/wasm/debug/build/gen/gen.component.wasm
	wasm-tools print $< > $@

main: target/wasm/debug/build/gen/gen.component.wasm
	python3 -m wasmtime.bindgen $< --out-dir $@

clean:
	rm -rf ffi gen interface target

run: main main.py
	python3 main.py
