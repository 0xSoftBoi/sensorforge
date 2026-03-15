import Foundation

/// Provides a unified clock for all sensor data.
///
/// All sensors record `ProcessInfo.processInfo.systemUptime` as their monotonic timestamp.
/// At session start we capture both `systemUptime` and `Date()` so we can convert
/// monotonic timestamps back to wall-clock time for export.
final class TimestampProvider {
    static let shared = TimestampProvider()

    private var anchorBootTime: TimeInterval = 0
    private var anchorWallClock: Date = Date()

    /// Call once at the start of each recording session.
    func anchor() {
        anchorBootTime = ProcessInfo.processInfo.systemUptime
        anchorWallClock = Date()
    }

    /// Current monotonic timestamp.
    var now: SensorTimestamp {
        SensorTimestamp(bootTime: ProcessInfo.processInfo.systemUptime)
    }

    /// Convert a monotonic boot time to a wall-clock date.
    func wallClock(from bootTime: TimeInterval) -> Date {
        let offset = bootTime - anchorBootTime
        return anchorWallClock.addingTimeInterval(offset)
    }

    /// Convert a monotonic boot time to seconds since session start.
    func sessionTime(from bootTime: TimeInterval) -> TimeInterval {
        bootTime - anchorBootTime
    }
}
