import Foundation
import os

/// Provides a unified clock for all sensor data.
///
/// All sensors record `ProcessInfo.processInfo.systemUptime` as their monotonic timestamp.
/// At session start we capture both `systemUptime` and `Date()` so we can convert
/// monotonic timestamps back to wall-clock time for export.
///
/// Thread-safe: all reads/writes to anchor values are protected by an `os_unfair_lock`.
final class TimestampProvider {
    static let shared = TimestampProvider()

    private var _lock = os_unfair_lock()
    private var _anchorBootTime: TimeInterval = 0
    private var _anchorWallClock: Date = Date()
    private var _isAnchored: Bool = false

    /// Whether `anchor()` has been called for the current session.
    var isAnchored: Bool {
        os_unfair_lock_lock(&_lock)
        defer { os_unfair_lock_unlock(&_lock) }
        return _isAnchored
    }

    /// Call once at the start of each recording session.
    func anchor() {
        os_unfair_lock_lock(&_lock)
        _anchorBootTime = ProcessInfo.processInfo.systemUptime
        _anchorWallClock = Date()
        _isAnchored = true
        os_unfair_lock_unlock(&_lock)
    }

    /// Reset anchor state (called when session ends).
    func reset() {
        os_unfair_lock_lock(&_lock)
        _isAnchored = false
        os_unfair_lock_unlock(&_lock)
    }

    /// Current monotonic timestamp.
    var now: SensorTimestamp {
        SensorTimestamp(bootTime: ProcessInfo.processInfo.systemUptime)
    }

    /// Convert a monotonic boot time to a wall-clock date.
    func wallClock(from bootTime: TimeInterval) -> Date {
        os_unfair_lock_lock(&_lock)
        let offset = bootTime - _anchorBootTime
        let wall = _anchorWallClock.addingTimeInterval(offset)
        os_unfair_lock_unlock(&_lock)
        return wall
    }

    /// Convert a monotonic boot time to seconds since session start.
    func sessionTime(from bootTime: TimeInterval) -> TimeInterval {
        os_unfair_lock_lock(&_lock)
        let result = bootTime - _anchorBootTime
        os_unfair_lock_unlock(&_lock)
        return result
    }
}
