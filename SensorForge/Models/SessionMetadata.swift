import Foundation

struct SessionMetadata: Identifiable, Codable {
    let id: UUID
    let startDate: Date
    var endDate: Date?
    var durationSeconds: Double?
    let deviceModel: String
    let osVersion: String

    // Sensor availability flags
    var hasLiDAR: Bool = false
    var hasARKit: Bool = false
    var hasGPS: Bool = false
    var hasBLE: Bool = false

    // Sample counts
    var frameCount: Int = 0
    var imuSampleCount: Int = 0
    var magnetometerSampleCount: Int = 0
    var gpsSampleCount: Int = 0
    var barometerSampleCount: Int = 0
    var bleTelemetryCount: Int = 0

    // Export info
    var exportDirectoryName: String
    var exportFormats: [String] = ["csv"]
    var totalFileSizeBytes: Int64?

    // Recording trigger
    var startMethod: StartMethod = .tap

    enum StartMethod: String, Codable {
        case tap
        case countdown
        case shake
        case siri
    }

    init(startMethod: StartMethod = .tap) {
        self.id = UUID()
        self.startDate = Date()
        self.deviceModel = Self.deviceModelName()
        self.osVersion = ProcessInfo.processInfo.operatingSystemVersionString
        self.exportDirectoryName = Self.directoryName(for: Date())
        self.startMethod = startMethod
    }

    private static func directoryName(for date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return "session_\(formatter.string(from: date))"
    }

    private static func deviceModelName() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machineMirror = Mirror(reflecting: systemInfo.machine)
        let identifier = machineMirror.children.reduce("") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return identifier }
            return identifier + String(UnicodeScalar(UInt8(value)))
        }
        return identifier
    }
}

/// Persists session metadata to disk as JSON.
final class SessionStore {
    static let shared = SessionStore()

    private let fileManager = FileManager.default

    var sessionsDirectory: URL {
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("SensorForge", isDirectory: true)
        try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private var metadataURL: URL {
        sessionsDirectory.appendingPathComponent("sessions.json")
    }

    func loadSessions() -> [SessionMetadata] {
        guard let data = try? Data(contentsOf: metadataURL) else { return [] }
        return (try? JSONDecoder().decode([SessionMetadata].self, from: data)) ?? []
    }

    func saveSessions(_ sessions: [SessionMetadata]) {
        let data = try? JSONEncoder().encode(sessions)
        try? data?.write(to: metadataURL, options: .atomic)
    }

    func sessionDirectory(for metadata: SessionMetadata) -> URL {
        let dir = sessionsDirectory.appendingPathComponent(metadata.exportDirectoryName, isDirectory: true)
        try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func deleteSession(_ metadata: SessionMetadata) {
        let dir = sessionDirectory(for: metadata)
        try? fileManager.removeItem(at: dir)

        var sessions = loadSessions()
        sessions.removeAll { $0.id == metadata.id }
        saveSessions(sessions)
    }

    func sessionSize(_ metadata: SessionMetadata) -> Int64 {
        let dir = sessionDirectory(for: metadata)
        guard let enumerator = fileManager.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += Int64(size)
            }
        }
        return total
    }
}
