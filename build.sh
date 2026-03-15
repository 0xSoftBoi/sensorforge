#!/bin/bash
# SensorForge Build Verification Script
# Run this on your Mac: ./build.sh
# Requires: Xcode 16+ installed

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$PROJECT_DIR/SensorForge.xcodeproj"
SCHEME="SensorForge"

echo "========================================="
echo " SensorForge Build Verification"
echo "========================================="
echo ""

# Check Xcode is available
if ! command -v xcodebuild &> /dev/null; then
    echo "ERROR: xcodebuild not found. Install Xcode from the App Store."
    exit 1
fi

XCODE_VERSION=$(xcodebuild -version | head -1)
echo "Using: $XCODE_VERSION"
echo "Project: $PROJECT"
echo ""

# Find an available simulator
SIMULATOR=$(xcrun simctl list devices available -j 2>/dev/null | \
    python3 -c "
import json, sys
data = json.load(sys.stdin)
for runtime, devices in data.get('devices', {}).items():
    if 'iOS' in runtime:
        for d in devices:
            if 'iPhone' in d['name'] and d['isAvailable']:
                print(d['name'])
                sys.exit(0)
print('iPhone 16 Pro')
" 2>/dev/null || echo "iPhone 16 Pro")

echo "Simulator: $SIMULATOR"
echo ""

# Build
echo "Building..."
echo "-----------------------------------------"

BUILD_LOG=$(mktemp)

if xcodebuild \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -sdk iphonesimulator \
    -destination "platform=iOS Simulator,name=$SIMULATOR" \
    -configuration Debug \
    build \
    2>&1 | tee "$BUILD_LOG"; then

    echo ""
    echo "========================================="
    echo " BUILD SUCCEEDED"
    echo "========================================="

    # Count warnings
    WARNINGS=$(grep -c "warning:" "$BUILD_LOG" 2>/dev/null || echo "0")
    echo "Warnings: $WARNINGS"

    if [ "$WARNINGS" -gt 0 ]; then
        echo ""
        echo "Warnings:"
        grep "warning:" "$BUILD_LOG" | head -20
    fi
else
    echo ""
    echo "========================================="
    echo " BUILD FAILED"
    echo "========================================="
    echo ""
    echo "Errors:"
    grep "error:" "$BUILD_LOG" | head -30
    echo ""
    echo "Full log: $BUILD_LOG"
    exit 1
fi

rm -f "$BUILD_LOG"
echo ""
echo "Done. Project compiles clean."
