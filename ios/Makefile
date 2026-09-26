# Jetlink for iPhone.
#
#   make project   regenerate Jetlink.xcodeproj from project.yml (needs xcodegen)
#   make build     compile for a device without signing, as CI would
#   make open      open the project in Xcode; pick a team, then run on a phone
SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

IOS_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
BUILD := $(IOS_DIR)/build

.PHONY: project build open clean

project:
	JETLINK_VERSION=0.0.0 JETLINK_BUILD=1 xcodegen generate --spec $(IOS_DIR)/project.yml --project $(IOS_DIR)

build: project
	xcodebuild build -project $(IOS_DIR)/Jetlink.xcodeproj -scheme Jetlink \
	  -destination 'generic/platform=iOS' -derivedDataPath $(BUILD)/DerivedData \
	  CODE_SIGNING_ALLOWED=NO

open: project
	open $(IOS_DIR)/Jetlink.xcodeproj

clean:
	rm -rf $(BUILD)
