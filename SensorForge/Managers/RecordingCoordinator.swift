import Foundation
import Combine

/// Orchestrates all sensor managers, providing a single start/stop interface
/// with synchronized timestamps.
@MainActor
final class RecordingCoordinator: ObservableObject {
    @Published var isRecording = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var sampleCount: Int = 0
    @Published var currentSession: SessionMetadata?

    let arkitManager = ARKitManager()
    let motionManager = MotionManager()
    let locationManager = LocationManager()
    let barometerManager = BarometerManager()
    let audioCaptureManager = AudioCaptureManager()
    let bleBridge = BLEBridge()

    let dataStore = SensorDataStore()

    private var durationTimer: Timer?
    private var recordingStartTime: TimeInterval = 0

    func startRecording(method: SessionMetadata.StartMethod = .tap) {
        guard !isRecording else { return }

        // Anchor the shared clock
        TimestampProvider.shared.anchor()
        recordingStartTime = ProcessInfo.processInfo.systemUptime

        // Create session metadata
        var session = SessionMetadata(startMethod: method)
        session.hasLiDAR = ARKitManager.isLiDARAvailable
        session.hasARKit = ARKitManager.isSupported
        session.hasGPS = true
        session.hasBLE = !bleBridge.connectedDevices.isEmpty
        currentSession = session

        // Create session directory and validate it exists
        let sessionDir = SessionStore.shared.sessionDirectory(for: session)
        guard FileManager.default.fileExists(atPath: sessionDir.path) else {
            print("[RecordingCoordinator] Failed to create session directory: \(sessionDir.path)")
            return
        }

        // Reset data store
        dataStore.reset()

        // Start all sensors
        arkitManager.start(dataStore: dataStore, sessionDirectory: sessionDir)
        motionManager.start(dataStore: dataStore)
        locationManager.start(dataStore: dataStore)
        barometerManager.start(dataStore: dataStore)
        audioCaptureManager.start(dataStore: dataStore, sessionDirectory: sessionDir)
        // BLE bridge continues if already scanning

        isRecording = true

        // Duration timer — created AFTER all sensors start successfully
        durationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.recordingDuration = ProcessInfo.processInfo.systemUptime - self.recordingStartTime
                self.sampleCount = self.dataStore.totalSampleCount
            }
        }
    }

    func stopRecording() {
        guard isRecording else { return }

        // Stop all sensors
        arkitManager.stop()
        motionManager.stop()
        locationManager.stop()
        barometerManager.stop()
        audioCaptureManager.stop()

        durationTimer?.invalidate()
        durationTimer = nil

        isRecording = false

        // Finalize session metadata
        guard var session = currentSession else { return }
        session.endDate = Date()
        session.durationSeconds = recordingDuration
        session.frameCount = dataStore.frames.count
        session.imuSampleCount = dataStore.imuSamples.count
        session.magnetometerSampleCount = dataStore.magnetometerSamples.count
        session.gpsSampleCount = dataStore.gpsSamples.count
        session.barometerSampleCount = dataStore.barometerSamples.count
        session.bleTelemetryCount = dataStore.bleTelemetry.count

        // Export to CSV
        let sessionDir = SessionStore.shared.sessionDirectory(for: session)
        CSVExporter.export(dataStore: dataStore, metadata: session, to: sessionDir)

        // Calculate size and save
        session.totalFileSizeBytes = SessionStore.shared.sessionSize(session)
        currentSession = session

        var sessions = SessionStore.shared.loadSessions()
        sessions.insert(session, at: 0)
        SessionStore.shared.saveSessions(sessions)
    }

    func requestPermissions() {
        locationManager.requestPermission()
    }
}
