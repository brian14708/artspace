.PHONY: prepare build dev lint fix env macos-post-bundle

ifeq ($(OS),Windows_NT) 
export PATH := $(shell pwd)/vendor/lib;${PATH}
else ifeq ($(shell uname -s),Darwin)
export DYLD_LIBRARY_PATH := $(shell pwd)/vendor/lib:${DYLD_LIBRARY_PATH}
else
export LD_LIBRARY_PATH := $(shell pwd)/vendor/lib:${LD_LIBRARY_PATH}
endif

ifdef CI
	CARGO_ARG := --release
else
	CARGO_ARG :=
endif

prepare: vendor/.prepare

build: prepare
	@pnpm recursive --filter ui exec tauri build

dev: prepare
	@pnpm recursive --filter ui exec tauri dev

test: prepare
	@cargo test ${CARGO_ARG}

lint: prepare
	@pnpm recursive run check
	@pnpm recursive run lint
	@cargo fmt --all -- --check
	@cargo clippy ${CARGO_ARG}

fix: prepare
	@pnpm recursive run format
	@cargo fmt
	@cargo clippy --fix

env: prepare
ifeq ($(OS),Windows_NT) 
	@echo "PATH=${PATH}"
else ifeq ($(shell uname -s),Darwin)
	@echo "DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}"
else
	@echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
endif

vendor/.prepare: scripts/prepare.py
	@python3 scripts/prepare.py
	@pnpm i
ifndef CI
ifeq ($(shell uname -s),Darwin)
	@mkdir -p .cargo
	@echo "[target.'cfg(target_os = \"macos\")']\nrustflags = [\"-C\", \"link-args=-Wl,-rpath,$(shell pwd)/vendor/lib,-rpath,@executable_path/../lib\"]" > .cargo/config.toml
endif
endif

macos-post-bundle:
ifdef RELEASE
	mkdir -p target/universal-apple-darwin/release/bundle/macos/artspace.app/Contents/lib/
	cp ./vendor/lib/libonnxruntime.1.12.0.dylib target/universal-apple-darwin/release/bundle/macos/artspace.app/Contents/lib/
	install_name_tool target/universal-apple-darwin/release/bundle/macos/artspace.app/Contents/MacOS/artspace -change @rpath/libonnxruntime.1.12.0.dylib @executable_path/../lib/libonnxruntime.1.12.0.dylib
else
	mkdir -p target/release/bundle/macos/artspace.app/Contents/lib/
	cp ./vendor/lib/libonnxruntime.1.12.0.dylib target/release/bundle/macos/artspace.app/Contents/lib/
	install_name_tool target/release/bundle/macos/artspace.app/Contents/MacOS/artspace -change @rpath/libonnxruntime.1.12.0.dylib @executable_path/../lib/libonnxruntime.1.12.0.dylib
endif
