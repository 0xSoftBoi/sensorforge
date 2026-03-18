import Foundation
import Network
import Combine
import os

/// WiFi bridge client — discovers Jetson via Bonjour, connects over TCP,
/// streams decimated sensor data, and receives robot/Qualia status.
///
/// Protocol: 4-byte big-endian length prefix + UTF-8 JSON payload.
/// Discovery: _sensorforge._tcp Bonjour service.
final class WiFiBridge: ObservableObject {

    // MARK: - Published State

    @Published var connectionState: ConnectionState = .disconnected
    @Published var jetsonName: String = ""
    @Published var qualiaActive: Bool = false
    @Published var lastScene: String = ""
    @Published var lastDirective: String = ""
    @Published var layerStates: [LayerState] = []
    @Published var voiceState: String = "idle"
    @Published var framesSent: Int = 0
    @Published var clockOffsetMs: Double = 0

    enum ConnectionState: String {
        case disconnected = "Disconnected"
        case browsing = "Searching..."
        case connecting = "Connecting..."
        case connected = "Connected"
        case error = "Error"
    }

    struct LayerState: Identifiable {
        let id: Int
        var vfe: Float
        var compression: Int
        var challenged: Bool
    }

    // MARK: - Private

    private let logger = Logger(subsystem: "com.sensorforge.app", category: "WiFiBridge")
    private var browser: NWBrowser?
    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "wifi-bridge", qos: .userInitiated)
    private var sendTimer: DispatchSourceTimer?
    private var clockOffsetNs: Int64 = 0

    // Sensor data snapshot (written by CaptureEngine, read by send timer)
    private let sensorLock = NSLock()
    private var latestIMU: (accel: SIMD3<Double>, gyro: SIMD3<Double>, mag: SIMD3<Double>)?
    private var latestPose: (pos: SIMD3<Float>, rot: simd_quatf, state: String)?
    private var latestGPS: (lat: Double, lon: Double, alt: Double, acc: Double)?
    private var latestBaro: (hpa: Double, alt: Double)?
    private var latestLight: Double = 0
    private var frameSeq: Int = 0

    static let port: UInt16 = 9876
    static let serviceType = "_sensorforge._tcp"

    // MARK: - Public API

    func startBrowsing() {
        guard connectionState == .disconnected || connectionState == .error else { return }
        DispatchQueue.main.async { self.connectionState = .browsing }

        let params = NWParameters()
        let browser = NWBrowser(for: .bonjour(type: Self.serviceType, domain: nil), using: params)

        browser.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.logger.info("Bonjour browser ready")
            case .failed(let error):
                self?.logger.error("Browser failed: \(error)")
                DispatchQueue.main.async { self?.connectionState = .error }
            default:
                break
            }
        }

        browser.browseResultsChangedHandler = { [weak self] results, _ in
            guard let self, let result = results.first else { return }
            self.logger.info("Found service: \(result.endpoint)")
            self.browser?.cancel()
            self.browser = nil
            self.connect(to: result.endpoint)
        }

        browser.start(queue: queue)
        self.browser = browser
    }

    func disconnect() {
        sendTimer?.cancel()
        sendTimer = nil
        connection?.cancel()
        connection = nil
        browser?.cancel()
        browser = nil
        DispatchQueue.main.async {
            self.connectionState = .disconnected
            self.jetsonName = ""
            self.qualiaActive = false
        }
    }

    func sendCommand(action: String, params: [String: Any] = [:]) {
        let msg: [String: Any] = ["type": "command", "action": action, "params": params]
        send(msg)
    }

    func requestStatus() {
        send(["type": "status_request"])
    }

    // MARK: - Sensor Data Injection (called by CaptureEngine)

    func updateIMU(accel: SIMD3<Double>, gyro: SIMD3<Double>, mag: SIMD3<Double>) {
        sensorLock.lock()
        latestIMU = (accel, gyro, mag)
        sensorLock.unlock()
    }

    func updatePose(position: SIMD3<Float>, rotation: simd_quatf, trackingState: String) {
        sensorLock.lock()
        latestPose = (position, rotation, trackingState)
        sensorLock.unlock()
    }

    func updateGPS(lat: Double, lon: Double, alt: Double, accuracy: Double) {
        sensorLock.lock()
        latestGPS = (lat, lon, alt, accuracy)
        sensorLock.unlock()
    }

    func updateBarometer(pressureHPa: Double, relativeAltitude: Double) {
        sensorLock.lock()
        latestBaro = (pressureHPa, relativeAltitude)
        sensorLock.unlock()
    }

    func updateAmbientLight(_ lux: Double) {
        sensorLock.lock()
        latestLight = lux
        sensorLock.unlock()
    }

    // MARK: - Connection

    private func connect(to endpoint: NWEndpoint) {
        DispatchQueue.main.async { self.connectionState = .connecting }

        let params = NWParameters.tcp
        let conn = NWConnection(to: endpoint, using: params)

        conn.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.logger.info("Connected to Jetson")
                self.sendHandshake()
                self.startReceiving()
                self.startSendTimer()
                DispatchQueue.main.async { self.connectionState = .connected }
            case .failed(let error):
                self.logger.error("Connection failed: \(error)")
                DispatchQueue.main.async { self.connectionState = .error }
            case .cancelled:
                DispatchQueue.main.async { self.connectionState = .disconnected }
            default:
                break
            }
        }

        conn.start(queue: queue)
        self.connection = conn
    }

    // MARK: - Handshake

    private func sendHandshake() {
        let deviceName = UIDevice.current.name
        let deviceId = UIDevice.current.identifierForVendor?.uuidString ?? UUID().uuidString
        let msg: [String: Any] = [
            "type": "handshake",
            "device_name": deviceName,
            "device_id": deviceId,
            "protocol_version": 1,
            "sensors_available": ["imu", "pose", "gps", "barometer", "ambient_light"],
            "timestamp_ns": SyncClock.shared.nowNanos,
        ]
        send(msg)
    }

    // MARK: - Send Timer (10Hz sensor frames)

    private func startSendTimer() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(100))
        timer.setEventHandler { [weak self] in
            self?.sendSensorFrame()
        }
        timer.resume()
        self.sendTimer = timer
    }

    private func sendSensorFrame() {
        sensorLock.lock()
        let imu = latestIMU
        let pose = latestPose
        let gps = latestGPS
        let baro = latestBaro
        let light = latestLight
        sensorLock.unlock()

        frameSeq += 1

        var msg: [String: Any] = [
            "type": "sensor_frame",
            "seq": frameSeq,
            "timestamp_ns": SyncClock.shared.nowNanos,
            "ambient_light": light,
        ]

        if let imu {
            msg["imu"] = [
                "accel": [imu.accel.x, imu.accel.y, imu.accel.z],
                "gyro": [imu.gyro.x, imu.gyro.y, imu.gyro.z],
                "mag": [imu.mag.x, imu.mag.y, imu.mag.z],
            ]
        }

        if let pose {
            msg["pose"] = [
                "position": [pose.pos.x, pose.pos.y, pose.pos.z],
                "rotation": [pose.rot.vector.x, pose.rot.vector.y, pose.rot.vector.z, pose.rot.vector.w],
                "tracking_state": pose.state,
            ]
        }

        if let gps {
            msg["gps"] = [
                "lat": gps.lat, "lon": gps.lon,
                "alt": gps.alt, "accuracy": gps.acc,
            ]
        }

        if let baro {
            msg["barometer"] = [
                "pressure_hpa": baro.hpa,
                "relative_altitude_m": baro.alt,
            ]
        }

        send(msg)
        DispatchQueue.main.async { self.framesSent = self.frameSeq }
    }

    // MARK: - Receive

    private func startReceiving() {
        receiveNextMessage()
    }

    private func receiveNextMessage() {
        guard let conn = connection else { return }

        // Read 4-byte length header
        conn.receive(minimumIncompleteLength: 4, maximumLength: 4) { [weak self] data, _, _, error in
            guard let self, let data, error == nil else { return }

            let length = data.withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
            guard length < 1_000_000 else { return }

            // Read payload
            conn.receive(minimumIncompleteLength: Int(length), maximumLength: Int(length)) { [weak self] data, _, _, error in
                guard let self, let data, error == nil else { return }
                self.handleMessage(data)
                self.receiveNextMessage()
            }
        }
    }

    private func handleMessage(_ data: Data) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }

        switch type {
        case "handshake_ack":
            let name = json["server_name"] as? String ?? "Jetson"
            let active = json["qualia_active"] as? Bool ?? false
            let offset = json["clock_offset_ns"] as? Int64 ?? 0
            clockOffsetNs = offset
            DispatchQueue.main.async {
                self.jetsonName = name
                self.qualiaActive = active
                self.clockOffsetMs = Double(offset) / 1_000_000
            }
            logger.info("Handshake ACK: \(name), qualia=\(active)")

        case "status_response":
            handleStatusResponse(json)

        case "command_ack":
            let action = json["action"] as? String ?? ""
            let success = json["success"] as? Bool ?? false
            let result = json["result"] as? String ?? ""
            logger.info("Command ACK: \(action) success=\(success) result=\(result)")

        case "clock_sync":
            if let serverNs = json["server_ns"] as? Int64 {
                let clientNs = json["client_ns"] as? Int64 ?? 0
                let rtt = Int64(SyncClock.shared.nowNanos) - clientNs
                clockOffsetNs = serverNs - clientNs - rtt / 2
                DispatchQueue.main.async {
                    self.clockOffsetMs = Double(self.clockOffsetNs) / 1_000_000
                }
            }

        default:
            logger.warning("Unknown message type: \(type)")
        }
    }

    private func handleStatusResponse(_ json: [String: Any]) {
        let vs = json["voice_state"] as? String ?? "idle"

        DispatchQueue.main.async {
            self.voiceState = vs
        }

        if let q = json["qualia"] as? [String: Any] {
            let active = q["active"] as? Bool ?? false
            let scene = q["scene"] as? String ?? ""
            let directive = q["directive"] as? String ?? ""

            var layers: [LayerState] = []
            if let layerArray = q["layers"] as? [[String: Any]] {
                for l in layerArray {
                    layers.append(LayerState(
                        id: l["id"] as? Int ?? 0,
                        vfe: (l["vfe"] as? NSNumber)?.floatValue ?? 0,
                        compression: l["compression"] as? Int ?? 0,
                        challenged: l["challenged"] as? Bool ?? false
                    ))
                }
            }

            DispatchQueue.main.async {
                self.qualiaActive = active
                self.lastScene = scene
                self.lastDirective = directive
                self.layerStates = layers
            }
        }
    }

    // MARK: - Send Helpers

    private func send(_ msg: [String: Any]) {
        guard let conn = connection else { return }
        guard let payload = try? JSONSerialization.data(withJSONObject: msg) else { return }

        var length = UInt32(payload.count).bigEndian
        var frame = Data(bytes: &length, count: 4)
        frame.append(payload)

        conn.send(content: frame, completion: .contentProcessed { [weak self] error in
            if let error {
                self?.logger.error("Send error: \(error)")
            }
        })
    }
}
