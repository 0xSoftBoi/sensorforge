import Foundation
import ARKit
import AVFoundation
import CoreMotion
import CoreLocation
import Combine

/// Records all iPhone sensors simultaneously, synchronized via SyncClock.
/// Outputs: video.mp4, audio.wav, imu.csv, gps.csv, magnetometer.csv,
/// barometer.csv, poses.csv, surfaces.jsonl, depth/*.bin, session.json
final class CaptureEngine: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var isRecording = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var frameCount: Int = 0
    @Published var storageUsedMB: Double = 0
    @Published var imuActive = false
    @Published var gpsActive = false
    @Published var depthActive = false
    @Published var baroActive = false
    @Published var audioActive = false

    // MARK: - Sensor Managers

    private let motionManager = CMMotionManager()
    private let altimeter = CMAltimeter()
    private let locationManager = CLLocationManager()

    private var arSession: ARSession?
    private var audioEngine: AVAudioEngine?
    private var audioFile: AVAudioFile?

    // MARK: - Video Writer

    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var videoStartTime: CMTime?

    // MARK: - File Handles

    private var imuHandle: FileHandle?
    private var gpsHandle: FileHandle?
    private var magHandle: FileHandle?
    private var baroHandle: FileHandle?
    private var poseHandle: FileHandle?
    private var surfaceHandle: FileHandle?

    // MARK: - Session State

    private var sessionDir: URL?
    private var depthDir: URL?
    private var depthFrameIndex: Int = 0
    private var clock: SyncClock!
    private var durationTimer: Timer?
    private var bleBridge: BLEBridge?

    // MARK: - Queues

    private let writeQueue = DispatchQueue(label: "com.sensorforge.write", qos: .userInitiated)
    private let imuQueue = OperationQueue()

    // MARK: - Init

    override init() {
        super.init()
        imuQueue.name = "com.sensorforge.imu"
        imuQueue.maxConcurrentOperationCount = 1
        locationManager.delegate = self
    }

    func setBLEBridge(_ bridge: BLEBridge) {
        self.bleBridge = bridge
    }

    // MARK: - Start Recording

    func startRecording() {
        guard !isRecording else { return }

        clock = SyncClock.newSession()
        frameCount = 0
        depthFrameIndex = 0
        storageUsedMB = 0
        recordingDuration = 0
        videoStartTime = nil

        // Create session directory
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let dir = docs.appendingPathComponent("sessions/\(timestamp)")
        let depthPath = dir.appendingPathComponent("depth")

        do {
            try FileManager.default.createDirectory(at: depthPath, withIntermediateDirectories: true)
        } catch {
            print("Failed to create session directory: \(error)")
            return
        }

        sessionDir = dir
        depthDir = depthPath

        // Open CSV files
        openCSVFiles(in: dir)

        // Set BLE bridge output
        bleBridge?.setOutputDirectory(dir)

        // Start all sensors
        startARSession()
        startIMU()
        startGPS()
        startBarometer()
        startAudio(in: dir)

        isRecording = true

        // Duration timer
        durationTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.recordingDuration = self.clock.elapsedSeconds
            self.updateStorageUsed()
        }
    }

    // MARK: - Stop Recording

    func stopRecording() {
        guard isRecording else { return }
        isRecording = false
        durationTimer?.invalidate()
        durationTimer = nil

        // Stop sensors
        arSession?.pause()
        arSession = nil
        motionManager.stopDeviceMotionUpdates()
        motionManager.stopMagnetometerUpdates()
        locationManager.stopUpdatingLocation()
        altimeter.stopRelativeAltitudeUpdates()

        // Stop audio
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
        audioFile = nil

        // Finalize video
        videoInput?.markAsFinished()
        assetWriter?.finishWriting { [weak self] in
            if let error = self?.assetWriter?.error {
                print("AssetWriter error: \(error)")
            }
        }

        // Close file handles
        closeAllHandles()

        // Write session metadata
        writeSessionMetadata()

        // Stop BLE logging
        bleBridge?.stopLogging()

        // Reset published state
        DispatchQueue.main.async {
            self.imuActive = false
            self.gpsActive = false
            self.depthActive = false
            self.baroActive = false
            self.audioActive = false
        }
    }

    // MARK: - ARSession

    private func startARSession() {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity

        // Enable LiDAR depth if available
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }

        // Enable scene reconstruction for surface classification
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
            config.sceneReconstruction = .meshWithClassification
        }

        let session = ARSession()
        session.delegate = self
        session.run(config)
        arSession = session
    }

    // MARK: - IMU (CoreMotion)

    private func startIMU() {
        guard motionManager.isDeviceMotionAvailable else { return }

        // Device motion at 100Hz (accel + gyro + gravity + attitude)
        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0
        motionManager.startDeviceMotionUpdates(using: .xArbitraryCorrectedZVertical, to: imuQueue) { [weak self] motion, error in
            guard let self, let motion, self.isRecording else { return }
            let t = self.clock.nanosFromBootTime(motion.timestamp)
            let a = motion.userAcceleration
            let g = motion.gravity
            let r = motion.rotationRate
            let q = motion.attitude.quaternion
            let line = "\(t),\(a.x),\(a.y),\(a.z),\(r.x),\(r.y),\(r.z),\(g.x),\(g.y),\(g.z),\(q.x),\(q.y),\(q.z),\(q.w)\n"
            self.appendToHandle(self.imuHandle, line)
            DispatchQueue.main.async { self.imuActive = true }
        }

        // Magnetometer at 50Hz
        if motionManager.isMagnetometerAvailable {
            motionManager.magnetometerUpdateInterval = 1.0 / 50.0
            motionManager.startMagnetometerUpdates(to: imuQueue) { [weak self] data, error in
                guard let self, let data, self.isRecording else { return }
                let t = self.clock.nanosFromBootTime(data.timestamp)
                let f = data.magneticField
                let line = "\(t),\(f.x),\(f.y),\(f.z)\n"
                self.appendToHandle(self.magHandle, line)
            }
        }
    }

    // MARK: - GPS

    private func startGPS() {
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    // MARK: - Barometer

    private func startBarometer() {
        guard CMAltimeter.isRelativeAltitudeAvailable() else { return }
        altimeter.startRelativeAltitudeUpdates(to: .main) { [weak self] data, error in
            guard let self, let data, self.isRecording else { return }
            let t = self.clock.nowNanos
            let pressure = data.pressure.doubleValue  // kPa
            let relAlt = data.relativeAltitude.doubleValue  // meters
            let line = "\(t),\(pressure),\(relAlt)\n"
            self.appendToHandle(self.baroHandle, line)
            self.baroActive = true
        }
    }

    // MARK: - Audio

    private func startAudio(in dir: URL) {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        let audioURL = dir.appendingPathComponent("audio.wav")

        do {
            let file = try AVAudioFile(forWriting: audioURL,
                                        settings: [
                                            AVFormatIDKey: kAudioFormatLinearPCM,
                                            AVSampleRateKey: 48000.0,
                                            AVNumberOfChannelsKey: format.channelCount,
                                            AVLinearPCMBitDepthKey: 16,
                                            AVLinearPCMIsFloatKey: false
                                        ])
            audioFile = file

            inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
                guard let self, self.isRecording else { return }
                do {
                    try self.audioFile?.write(from: buffer)
                } catch {
                    print("Audio write error: \(error)")
                }
            }

            try engine.start()
            audioEngine = engine
            DispatchQueue.main.async { self.audioActive = true }
        } catch {
            print("Audio setup error: \(error)")
        }
    }

    // MARK: - CSV File Management

    private func openCSVFiles(in dir: URL) {
        let files: [(String, String, inout FileHandle?)] = [
            ("imu.csv", "timestamp_ns,ax,ay,az,gx,gy,gz,gravx,gravy,gravz,qx,qy,qz,qw\n", &imuHandle),
            ("gps.csv", "timestamp_ns,lat,lon,alt,hacc,vacc,speed,course\n", &gpsHandle),
            ("magnetometer.csv", "timestamp_ns,mx,my,mz\n", &magHandle),
            ("barometer.csv", "timestamp_ns,pressure_kpa,rel_alt_m\n", &baroHandle),
            ("poses.csv", "timestamp_ns,x,y,z,qx,qy,qz,qw,pitch,yaw,roll,tracking\n", &poseHandle),
            ("surfaces.jsonl", "", &surfaceHandle),
        ]

        // We can't use inout with tuples in a loop, so do them individually
        createCSV(dir, "imu.csv", "timestamp_ns,ax,ay,az,gx,gy,gz,gravx,gravy,gravz,qx,qy,qz,qw\n", &imuHandle)
        createCSV(dir, "gps.csv", "timestamp_ns,lat,lon,alt,hacc,vacc,speed,course\n", &gpsHandle)
        createCSV(dir, "magnetometer.csv", "timestamp_ns,mx,my,mz\n", &magHandle)
        createCSV(dir, "barometer.csv", "timestamp_ns,pressure_kpa,rel_alt_m\n", &baroHandle)
        createCSV(dir, "poses.csv", "timestamp_ns,x,y,z,qx,qy,qz,qw,pitch,yaw,roll,tracking\n", &poseHandle)
        createCSV(dir, "surfaces.jsonl", "", &surfaceHandle)
    }

    private func createCSV(_ dir: URL, _ name: String, _ header: String, _ handle: inout FileHandle?) {
        let url = dir.appendingPathComponent(name)
        FileManager.default.createFile(atPath: url.path, contents: header.data(using: .utf8))
        handle = try? FileHandle(forWritingTo: url)
        handle?.seekToEndOfFile()
    }

    private func closeAllHandles() {
        for handle in [imuHandle, gpsHandle, magHandle, baroHandle, poseHandle, surfaceHandle] {
            try? handle?.close()
        }
        imuHandle = nil
        gpsHandle = nil
        magHandle = nil
        baroHandle = nil
        poseHandle = nil
        surfaceHandle = nil
    }

    private func appendToHandle(_ handle: FileHandle?, _ string: String) {
        guard let handle, let data = string.data(using: .utf8) else { return }
        writeQueue.async {
            handle.write(data)
        }
    }

    // MARK: - Video Writing

    private func setupVideoWriter(pixelBuffer: CVPixelBuffer) {
        guard let dir = sessionDir else { return }
        let videoURL = dir.appendingPathComponent("video.mp4")

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        do {
            let writer = try AVAssetWriter(url: videoURL, fileType: .mp4)

            let settings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.hevc,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: 10_000_000,  // 10 Mbps
                    AVVideoExpectedSourceFrameRateKey: 30,
                    AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main_AutoLevel,
                ]
            ]

            let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
            input.expectsMediaDataInRealTime = true

            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: width,
                    kCVPixelBufferHeightKey as String: height,
                ]
            )

            writer.add(input)
            writer.startWriting()

            assetWriter = writer
            videoInput = input
            pixelBufferAdaptor = adaptor
        } catch {
            print("Video writer setup error: \(error)")
        }
    }

    // MARK: - Depth Writing

    private func writeDepthFrame(_ depthMap: CVPixelBuffer) {
        guard let depthDir else { return }
        let index = depthFrameIndex
        depthFrameIndex += 1

        writeQueue.async {
            CVPixelBufferLockBaseAddress(depthMap, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

            let width = CVPixelBufferGetWidth(depthMap)
            let height = CVPixelBufferGetHeight(depthMap)
            guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return }

            let byteCount = width * height * MemoryLayout<Float32>.size
            let data = Data(bytes: baseAddress, count: byteCount)

            let filename = String(format: "%06d.bin", index)
            let url = depthDir.appendingPathComponent(filename)
            try? data.write(to: url)
        }
    }

    // MARK: - Session Metadata

    private func writeSessionMetadata() {
        guard let dir = sessionDir else { return }

        let device = UIDevice.current
        let metadata: [String: Any] = [
            "app": "SensorForge",
            "version": "0.1.0",
            "session_start": clock.startISO,
            "session_end": ISO8601DateFormatter().string(from: Date()),
            "duration_seconds": clock.elapsedSeconds,
            "frame_count": frameCount,
            "device": [
                "model": device.model,
                "name": device.name,
                "system_name": device.systemName,
                "system_version": device.systemVersion,
            ],
            "sensors": [
                "imu_hz": 100,
                "magnetometer_hz": 50,
                "gps_hz": 1,
                "barometer_hz": 1,
                "video_fps": 30,
                "video_codec": "hevc",
                "audio_sample_rate": 48000,
                "depth_resolution": "256x192",
            ],
            "clock": "mach_absolute_time_ns",
        ]

        let url = dir.appendingPathComponent("session.json")
        if let data = try? JSONSerialization.data(withJSONObject: metadata, options: .prettyPrinted) {
            try? data.write(to: url)
        }
    }

    // MARK: - Storage

    private func updateStorageUsed() {
        guard let dir = sessionDir else { return }
        writeQueue.async { [weak self] in
            let enumerator = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey])
            var total: Int64 = 0
            while let url = enumerator?.nextObject() as? URL {
                if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    total += Int64(size)
                }
            }
            DispatchQueue.main.async {
                self?.storageUsedMB = Double(total) / (1024 * 1024)
            }
        }
    }

    /// Returns the URL of the most recent session directory for sharing.
    var lastSessionURL: URL? { sessionDir }
}

// MARK: - ARSessionDelegate

extension CaptureEngine: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isRecording else { return }
        let t = clock.nowNanos

        // --- Video frame ---
        let pixelBuffer = frame.capturedImage

        if assetWriter == nil {
            setupVideoWriter(pixelBuffer: pixelBuffer)
        }

        if let writer = assetWriter, writer.status == .readyForMoreMediaData || writer.status == .writing {
            let frameTime = CMTime(value: Int64(t), timescale: 1_000_000_000)
            if videoStartTime == nil {
                videoStartTime = frameTime
                writer.startSession(atSourceTime: frameTime)
            }
            if let adaptor = pixelBufferAdaptor, let input = videoInput, input.isReadyForMoreMediaData {
                adaptor.append(pixelBuffer, withPresentationTime: frameTime)
                DispatchQueue.main.async { self.frameCount += 1 }
            }
        }

        // --- Camera pose ---
        let transform = frame.camera.transform
        let pos = transform.columns.3
        let euler = frame.camera.eulerAngles
        let q = simd_quatf(transform)
        let trackingState: String
        switch frame.camera.trackingState {
        case .normal: trackingState = "normal"
        case .limited(let reason):
            switch reason {
            case .excessiveMotion: trackingState = "limited_motion"
            case .insufficientFeatures: trackingState = "limited_features"
            case .initializing: trackingState = "initializing"
            case .relocalizing: trackingState = "relocalizing"
            @unknown default: trackingState = "limited_unknown"
            }
        case .notAvailable: trackingState = "not_available"
        }
        let poseLine = "\(t),\(pos.x),\(pos.y),\(pos.z),\(q.imag.x),\(q.imag.y),\(q.imag.z),\(q.real),\(euler.x),\(euler.y),\(euler.z),\(trackingState)\n"
        appendToHandle(poseHandle, poseLine)

        // --- LiDAR depth ---
        if let depth = frame.sceneDepth?.depthMap {
            writeDepthFrame(depth)
            DispatchQueue.main.async { self.depthActive = true }
        }

        // --- Surface classifications ---
        if let anchors = session.currentFrame?.anchors {
            for anchor in anchors {
                if let meshAnchor = anchor as? ARMeshAnchor {
                    let classifications = meshAnchor.geometry.classification
                    if classifications.format != .invalid {
                        let entry: [String: Any] = [
                            "timestamp_ns": t,
                            "anchor_id": meshAnchor.identifier.uuidString,
                            "transform": [
                                meshAnchor.transform.columns.3.x,
                                meshAnchor.transform.columns.3.y,
                                meshAnchor.transform.columns.3.z
                            ],
                        ]
                        if let data = try? JSONSerialization.data(withJSONObject: entry),
                           let line = String(data: data, encoding: .utf8) {
                            appendToHandle(surfaceHandle, line + "\n")
                        }
                    }
                }
            }
        }
    }
}

// MARK: - CLLocationManagerDelegate

extension CaptureEngine: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard isRecording else { return }
        for loc in locations {
            let t = clock.nowNanos
            let line = "\(t),\(loc.coordinate.latitude),\(loc.coordinate.longitude),\(loc.altitude),\(loc.horizontalAccuracy),\(loc.verticalAccuracy),\(loc.speed),\(loc.course)\n"
            appendToHandle(gpsHandle, line)
            DispatchQueue.main.async { self.gpsActive = true }
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error)")
    }
}
