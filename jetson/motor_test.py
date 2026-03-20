#!/usr/bin/env python3
"""
Motor Diagnostic — standalone test to verify ESP32 + motors work.

Bypasses the full Qualia stack. Opens serial directly (no heartbeat thread,
no SHM, no threading races) and sends sustained commands for 3 seconds each.

Usage:
    python3 motor_test.py [--port /dev/ttyTHS1] [--speed 80]
"""

import json
import sys
import time


def send(ser, left, right):
    """Send a motor command and log it."""
    cmd = {"T": 11, "L": left, "R": right}
    line = json.dumps(cmd) + "\n"
    ser.write(line.encode())
    ser.flush()
    print(f"  TX: {line.strip()}")


def drain(ser):
    """Read and print any ESP32 responses."""
    responses = []
    while ser.in_waiting:
        try:
            data = ser.readline().decode(errors="replace").strip()
            if data:
                responses.append(data)
                print(f"  RX: {data}")
        except Exception:
            break
    return responses


def hold_command(ser, left, right, duration=3.0, resend_interval=0.5):
    """Send a motor command repeatedly for `duration` seconds (no heartbeat thread needed)."""
    end = time.monotonic() + duration
    while time.monotonic() < end:
        send(ser, left, right)
        drain(ser)
        time.sleep(resend_interval)


def run_test(ser, name, left, right, speed, duration=3.0):
    """Run one motor test and ask user for confirmation."""
    print(f"\n{'='*50}")
    print(f"TEST: {name}")
    print(f"  Command: L={left}, R={right}")
    print(f"  Duration: {duration}s")
    print(f"{'='*50}")

    input("  Press ENTER to start (motors will spin!)... ")

    hold_command(ser, left, right, duration)

    # Stop
    send(ser, 0, 0)
    time.sleep(0.2)
    drain(ser)

    result = input("  Did the wheels move correctly? [y/n/s(kip)]: ").strip().lower()
    if result == "y":
        return "PASS"
    elif result == "s":
        return "SKIP"
    return "FAIL"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Standalone motor diagnostic")
    parser.add_argument("--port", default="/dev/ttyTHS1", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--speed", type=int, default=80, help="Test speed (0-128)")
    args = parser.parse_args()

    speed = min(abs(args.speed), 128)

    print(f"Motor Diagnostic Tool")
    print(f"  Port:  {args.port}")
    print(f"  Baud:  {args.baud}")
    print(f"  Speed: {speed}")
    print()

    # Open serial directly — no threads, no heartbeat
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed. Run: pip install pyserial")
        sys.exit(1)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except Exception as e:
        print(f"ERROR: Cannot open {args.port}: {e}")
        print("  - Is the USB cable connected?")
        print("  - Is another process using the port? (check: fuser /dev/ttyTHS1)")
        print("  - Is ESP32 in bootloader mode? (try unplugging and replugging)")
        sys.exit(1)

    time.sleep(1.0)  # ESP32 reset delay after serial open
    print("Serial connected. Draining startup messages...")
    drain(ser)

    # Send initial stop
    print("Sending initial STOP...")
    send(ser, 0, 0)
    time.sleep(0.5)
    drain(ser)

    # Run tests
    results = {}

    tests = [
        ("FORWARD",    speed,  speed),
        ("BACKWARD",  -speed, -speed),
        ("LEFT TURN", -speed,  speed),
        ("RIGHT TURN", speed, -speed),
    ]

    for name, left, right in tests:
        results[name] = run_test(ser, name, left, right, speed)

    # Final stop
    send(ser, 0, 0)
    ser.close()

    # Summary
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    all_pass = True
    for name, result in results.items():
        icon = {"PASS": "+", "FAIL": "X", "SKIP": "-"}.get(result, "?")
        print(f"  [{icon}] {name}: {result}")
        if result == "FAIL":
            all_pass = False

    if all_pass:
        print("\nAll tests passed! Motors are working.")
        print("If the autonomous explorer still doesn't move, the bug is in software (heartbeat race).")
    else:
        print("\nSome tests FAILED. Possible causes:")
        print("  - Battery dead or motor power switch off")
        print("  - ESP32 not running motor firmware (check with Arduino IDE)")
        print("  - Wrong serial port (try /dev/ttyACM1 or /dev/ttyUSB0)")
        print("  - Motor wiring disconnected")


if __name__ == "__main__":
    main()
