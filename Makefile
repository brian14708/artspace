.PHONY: prepare build dev lint fix env \
	pre-build-macos post-build-macos \
	pre-build-linux post-build-linux \
	pre-build-windows post-build-windows

ifdef CI
RELEASE := 1
endif

VERBOSE ?= @
CARGO_ARGS :=
TAURI_ARGS :=
HOST_TARGET := $(shell rustc -vV | awk '/host/ { print $$2 }')
ifdef RELEASE
CARGO_ARGS += --release
endif
ifdef TARGET
CARGO_ARGS += --target "$(TARGET)"
TAURI_ARGS += --target "$(TARGET)"
else
TARGET := $(HOST_TARGET)
endif

ifeq ($(HOST_TARGET),$(TARGET))
ifdef RELEASE
OUT_DIR := target/release
else
OUT_DIR := target/debug
endif
else
ifdef RELEASE
OUT_DIR := target/$(TARGET)/release
else
OUT_DIR := target/$(TARGET)/debug
endif
endif

# set env for running with vendored library
ifeq ($(OS),Windows_NT) 
export PATH := $(shell pwd)/vendor/$(TARGET)/lib;${PATH}
else ifeq ($(shell uname -s),Darwin)
export DYLD_LIBRARY_PATH := $(shell pwd)/vendor/$(TARGET)/lib:${DYLD_LIBRARY_PATH}
else
export LD_LIBRARY_PATH := $(shell pwd)/vendor/$(TARGET)/lib:${LD_LIBRARY_PATH}
endif

ifeq ($(OS),Windows_NT)
TARGET_OS=windows
else
ifeq ($(shell uname -s),Darwin)
TARGET_OS=macos
else
TARGET_OS=linux
endif
endif

prepare: vendor/$(TARGET)/.prepare
ifdef CI
	$(VERBOSE)rm -f .cargo/config.toml
else
	$(VERBOSE)mkdir -p .cargo
	$(VERBOSE)echo "[target.'cfg(target_os = \"macos\")']" > .cargo/config.toml
	$(VERBOSE)echo "rustflags = [\"-C\", \"link-args=-Wl,-rpath,$(shell pwd)/vendor/$(TARGET)/lib,-rpath,@executable_path/../lib\"]" >> .cargo/config.toml
	$(VERBOSE)echo "[target.'cfg(target_os = \"linux\")']" >> .cargo/config.toml
	$(VERBOSE)echo "rustflags = [\"-C\", \"link-args=-Wl,-rpath,$(shell pwd)/vendor/$(TARGET)/lib\"]" >> .cargo/config.toml
endif

build: prepare
	$(VERBOSE)$(MAKE) RELEASE=1 pre-build-$(TARGET_OS)
	$(VERBOSE)pnpm recursive --filter app exec tauri build $(TAURI_ARGS)
	$(VERBOSE)$(MAKE) RELEASE=1 post-build-$(TARGET_OS)

dev: prepare
	$(VERBOSE)pnpm recursive --filter app exec tauri dev $(TAURI_ARGS)

test: prepare
	$(VERBOSE)cargo test $(CARGO_ARGS)

cli: prepare
	$(VERBOSE)cargo run $(CARGO_ARGS) -p artspace-cli -- $(ARGS)

lint: prepare
	$(VERBOSE)pnpm recursive run check
	$(VERBOSE)pnpm recursive run lint
	$(VERBOSE)cargo fmt --all -- --check
	$(VERBOSE)cargo clippy $(CARGO_ARGS)

fix: prepare
	$(VERBOSE)pnpm recursive run format
	$(VERBOSE)cargo fmt --all
	$(VERBOSE)cargo clippy --fix $(CARGO_ARGS)

env: prepare
	$(VERBOSE)echo "BUILD_TARGET=$(TARGET)"
	$(VERBOSE)echo "BUILD_OUT_DIR=$(OUT_DIR)"
ifeq ($(TARGET_OS),linux)
	$(VERBOSE)echo "LD_LIBRARY_PATH=$(LD_LIBRARY_PATH)"
endif

vendor/universal-apple-darwin/.prepare: vendor/aarch64-apple-darwin/.prepare
	$(VERBOSE)$(MAKE) vendor/x86_64-apple-darwin/.prepare
	$(VERBOSE)ln -s x86_64-apple-darwin vendor/universal-apple-darwin
	$(VERBOSE)touch $@

vendor/%/.prepare: scripts/prepare.py app/package.json
	$(VERBOSE)python3 scripts/prepare.py $@
	$(VERBOSE)pnpm i

pre-build-windows:
	$(VERBOSE)mkdir -p $(OUT_DIR)/deps
	$(VERBOSE)cp ./vendor/$(TARGET)/lib/onnxruntime*.dll $(OUT_DIR)
	$(VERBOSE)cp ./vendor/$(TARGET)/lib/onnxruntime*.dll $(OUT_DIR)/deps
post-build-windows:

pre-build-macos:
post-build-macos:
	$(VERBOSE)mkdir -p $(OUT_DIR)/bundle/macos/artspace.app/Contents/lib/
	$(VERBOSE)cp ./vendor/$(TARGET)/lib/libonnxruntime.1.12.1.dylib $(OUT_DIR)/bundle/macos/artspace.app/Contents/lib/
	$(VERBOSE)install_name_tool $(OUT_DIR)/bundle/macos/artspace.app/Contents/MacOS/artspace -change @rpath/libonnxruntime.1.12.1.dylib @executable_path/../lib/libonnxruntime.1.12.1.dylib

pre-build-linux:
	$(VERBOSE)mkdir -p $(OUT_DIR)/bundle/appimage_deb/data/usr/share/artspace/providers
	$(VERBOSE)cp ./vendor/$(TARGET)/lib/libonnxruntime_providers_*.so $(OUT_DIR)/bundle/appimage_deb/data/usr/share/artspace/providers
post-build-linux:
