import Foundation
import ARKit
import AVFoundation
import Combine

/// Manages the ARSession and captures frame data (camera pose, depth, mesh, planes, light).
final class ARKitManager: NSObject, ObservableObject {
    @Published var isRunning = false
    @Published var trackingState: String = "Not Available"
    @Published var hasLiDAR = false
    @Published var currentFPS: Double = 0

    private var session: ARSession?
    private var dataStore: SensorDataStore?
    private var sessionDirectory: URL?
    private var frameIndex = 0
    private var fpsCounter = 0
    private var fpsTimer: TimeInterval = 0

    // Serial queue protecting all video writer state and frame counters
    private let writerQueue = DispatchQueue(label: "com.sensorforge.arkit.writer")

    // Video writer — only accessed on writerQueue
    private var videoWriter: AVAssetWriter?
    private var videoWriterInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var videoStartTime: CMTime?

    static var isSupported: Bool {
        ARWorldTrackingConfiguration.isSupported
    }

    static var isLiDARAvailable: Bool {
        ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
    }

    func start(dataStore: SensorDataStore, sessionDirectory: URL) {
        self.dataStore = dataStore
        self.sessionDirectory = sessionDirectory
        self.frameIndex = 0
        self.hasLiDAR = Self.isLiDARAvailable

        let session = ARSession()
        session.delegate = self
        self.session = session

        let config = ARWorldTrackingConfiguration()

        // Enable LiDAR scene reconstruction if available
        if Self.isLiDARAvailable {
            config.sceneReconstruction = .meshWithClassification
            config.frameSemantics.insert(.sceneDepth)
        }

        // Enable plane detection
        config.planeDetection = [.horizontal, .vertical]

        // Enable light estimation
        config.isLightEstimationEnabled = true

        // High-res video if possible
        if let hiResFormat = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: {
            $0.imageResolution.width >= 3840
        }) {
            config.videoFormat = hiResFormat
        }

        session.run(config)
        DispatchQueue.main.async { self.isRunning = true }

        setupVideoWriter(sessionDirectory: sessionDirectory)
    }

    func stop() {
        session?.pause()
        session = nil
        DispatchQueue.main.async { self.isRunning = false }
        finalizeVideoWriter()
    }

    // MARK: - Video Writer

    private func setupVideoWriter(sessionDirectory: URL) {
        writerQueue.async { [weak self] in
            guard let self else { return }

            let videoURL = sessionDirectory.appendingPathComponent("rgb_video.mp4")

            guard let writer = try? AVAssetWriter(url: videoURL, fileType: .mp4) else {
                print("[ARKitManager] Failed to create video writer at \(videoURL)")
                return
            }

            let settings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.hevc,
                AVVideoWidthKey: 1920,
                AVVideoHeightKey: 1440,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: 20_000_000,
                    AVVideoExpectedSourceFrameRateKey: 30
                ]
            ]

            let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
            input.expectsMediaDataInRealTime = true

            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                ]
            )

            writer.add(input)
            writer.startWriting()

            self.videoWriter = writer
            self.videoWriterInput = input
            self.pixelBufferAdaptor = adaptor
        }
    }

    /// Must be called on writerQueue.
    private func writeVideoFrameOnQueue(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        guard let writer = videoWriter,
              let input = videoWriterInput,
              let adaptor = pixelBufferAdaptor,
              writer.status == .writing else { return }

        let cmTime = CMTime(seconds: timestamp, preferredTimescale: 600)

        if videoStartTime == nil {
            videoStartTime = cmTime
            writer.startSession(atSourceTime: cmTime)
        }

        if input.isReadyForMoreMediaData {
            adaptor.append(pixelBuffer, withPresentationTime: cmTime)
        }
    }

    private func finalizeVideoWriter() {
        writerQueue.async { [weak self] in
            guard let self else { return }
            guard let writer = self.videoWriter, writer.status == .writing else {
                self.videoWriter = nil
                self.videoWriterInput = nil
                self.pixelBufferAdaptor = nil
                self.videoStartTime = nil
                return
            }
            self.videoWriterInput?.markAsFinished()
            writer.finishWriting {
                // Cleanup on writerQueue to avoid races
                self.writerQueue.async {
                    self.videoWriter = nil
                    self.videoWriterInput = nil
                    self.pixelBufferAdaptor = nil
                    self.videoStartTime = nil
                }
            }
        }
    }
}

// MARK: - ARSessionDelegate

extension ARKitManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let ts = SensorTimestamp(bootTime: frame.timestamp)

        // FPS tracking — dispatch counter updates to writerQueue to avoid races
        writerQueue.async { [weak self] in
            guard let self else { return }
            self.fpsCounter += 1
            if frame.timestamp - self.fpsTimer >= 1.0 {
                let fps = Double(self.fpsCounter)
                DispatchQueue.main.async { self.currentFPS = fps }
                self.fpsCounter = 0
                self.fpsTimer = frame.timestamp
            }
            self.frameIndex += 1
        }

        // Tracking state
        let stateString: String
        switch frame.camera.trackingState {
        case .normal: stateString = "Normal"
        case .limited(let reason):
            switch reason {
            case .initializing: stateString = "Initializing"
            case .excessiveMotion: stateString = "Excessive Motion"
            case .insufficientFeatures: stateString = "Insufficient Features"
            case .relocalizing: stateString = "Relocalizing"
            @unknown default: stateString = "Limited"
            }
        case .notAvailable: stateString = "Not Available"
        }

        DispatchQueue.main.async { self.trackingState = stateString }

        // Light estimation
        let ambientIntensity = frame.lightEstimate?.ambientIntensity.map { Float($0) }
        let ambientTemp = frame.lightEstimate?.ambientColorTemperature.map { Float($0) }

        // Depth info
        var depthW: Int?
        var depthH: Int?
        if let depthMap = frame.sceneDepth?.depthMap {
            depthW = CVPixelBufferGetWidth(depthMap)
            depthH = CVPixelBufferGetHeight(depthMap)
        }

        let frameData = ARFrameData(
            timestamp: ts,
            cameraPose: frame.camera.transform,
            cameraIntrinsics: frame.camera.intrinsics,
            imageResolution: frame.camera.imageResolution,
            trackingState: stateString,
            ambientIntensity: ambientIntensity,
            ambientColorTemperature: ambientTemp,
            depthWidth: depthW,
            depthHeight: depthH
        )

        // Write video frame on the writer queue
        let capturedImage = frame.capturedImage
        let frameTimestamp = frame.timestamp
        writerQueue.async { [weak self] in
            self?.writeVideoFrameOnQueue(capturedImage, timestamp: frameTimestamp)
        }

        Task { @MainActor in
            self.dataStore?.frames.append(frameData)
        }
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        processAnchors(anchors)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        processAnchors(anchors)
    }

    private func processAnchors(_ anchors: [ARAnchor]) {
        let ts = TimestampProvider.shared.now

        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                let data = MeshAnchorData(
                    timestamp: ts,
                    identifier: meshAnchor.identifier,
                    transform: meshAnchor.transform,
                    vertexCount: meshAnchor.geometry.vertices.count,
                    faceCount: meshAnchor.geometry.faces.count,
                    classification: nil
                )
                Task { @MainActor in
                    self.dataStore?.meshAnchors.append(data)
                }
            } else if let planeAnchor = anchor as? ARPlaneAnchor {
                let classification: String
                switch planeAnchor.classification {
                case .wall: classification = "wall"
                case .floor: classification = "floor"
                case .ceiling: classification = "ceiling"
                case .table: classification = "table"
                case .seat: classification = "seat"
                case .door: classification = "door"
                case .window: classification = "window"
                case .none: classification = "none"
                case .notAvailable: classification = "notAvailable"
                @unknown default: classification = "unknown"
                }

                let data = PlaneAnchorData(
                    timestamp: ts,
                    identifier: planeAnchor.identifier,
                    transform: planeAnchor.transform,
                    classification: classification,
                    extentX: planeAnchor.extent.x,
                    extentZ: planeAnchor.extent.z
                )
                Task { @MainActor in
                    self.dataStore?.planeAnchors.append(data)
                }
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[ARKitManager] Session error: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        print("[ARKitManager] Session interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        print("[ARKitManager] Session interruption ended")
    }
}
