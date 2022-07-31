.PHONY: prepare build dev lint fix env

ifeq ($(OS),Windows_NT) 
export PATH := $(shell pwd)/vendor/lib:${PATH}
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
else
	@echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
endif

vendor/.prepare: scripts/prepare.py
	@python3 scripts/prepare.py
	@pnpm i
ifdef CI
ifeq ($(OS),Windows_NT) 
	@mkdir -p target/release
	@cp vendor/lib/*.dll ./target/release/
endif
endif
