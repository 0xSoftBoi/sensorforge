import Foundation
import QuartzCore

/// Provides nanosecond-precision monotonic timestamps for synchronizing all sensor readings.
/// Uses mach_absolute_time for consistent timing across all capture sources.
final class SyncClock {
    static let shared = SyncClock()

    private let timebaseInfo: mach_timebase_info_data_t
    private let sessionStartMach: UInt64
    private let sessionStartWall: Date

    private init() {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        self.timebaseInfo = info
        self.sessionStartMach = mach_absolute_time()
        self.sessionStartWall = Date()
    }

    /// Current monotonic timestamp in nanoseconds since boot.
    var nowNanos: UInt64 {
        let machTime = mach_absolute_time()
        return machTime * UInt64(timebaseInfo.numer) / UInt64(timebaseInfo.denom)
    }

    /// Elapsed nanoseconds since this session started.
    var elapsedNanos: UInt64 {
        let now = mach_absolute_time()
        let elapsed = now - sessionStartMach
        return elapsed * UInt64(timebaseInfo.numer) / UInt64(timebaseInfo.denom)
    }

    /// Elapsed seconds since session start, as a Double for CSV output.
    var elapsedSeconds: Double {
        Double(elapsedNanos) / 1_000_000_000.0
    }

    /// Wall-clock Date when the session started (for metadata).
    var startDate: Date { sessionStartWall }

    /// ISO 8601 string of the session start time.
    var startISO: String {
        ISO8601DateFormatter().string(from: sessionStartWall)
    }

    /// Convert a CMTime or CACurrentMediaTime()-based timestamp to our nanosecond clock.
    /// CACurrentMediaTime uses mach_absolute_time internally, so we can convert directly.
    func nanosFromMediaTime(_ mediaTime: CFTimeInterval) -> UInt64 {
        UInt64(mediaTime * 1_000_000_000)
    }

    /// Convert a CoreMotion timestamp (which is time since boot) to our nanos.
    func nanosFromBootTime(_ bootTime: TimeInterval) -> UInt64 {
        UInt64(bootTime * 1_000_000_000)
    }

    /// Create a new clock instance (call at start of each recording session).
    static func newSession() -> SyncClock {
        SyncClock()
    }
}
