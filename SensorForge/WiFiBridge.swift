import Foundation
import Combine
import Network

/// Connects to a Waveshare UGV (or similar ESP32 robot) over WiFi HTTP,
/// polls telemetry at 10Hz, and logs structured data to CSV.
///
/// The Waveshare ESP32 sub-controller speaks JSON commands of the form:
///   {"T": <command_type>, ...}
/// Telemetry is polled via:
///   {"T": 126} → IMU data (heading, accel, pitch, roll)
///   {"T": 130} → Chassis feedback (encoder speeds, battery)
final class WiFiBridge: ObservableObject {

    // MARK: - Published State

    @Published var isConnected = false
    @Published var isLogging = false
    @Published var hostAddress: String = "192.168.4.1"
    @Published var port: Int = 5000
    @Published var telemetryRate: Double = 0
    @Published var batteryVoltage: Double = 0
    @Published var messageCount: Int = 0

    // MARK: - Private

    private var pollTimer: Timer?
    private var heartbeatTimer: Timer?
    private var outputHandle: FileHandle?
    private let clock = SyncClock.shared
    private let writeQueue = DispatchQueue(label: "com.sensorforge.wifi.write", qos: .userInitiated)
    private let session: URLSession
    private var messagesThisSecond: Int = 0
    private var rateTimer: Timer?
    private var monitor: NWPathMonitor?

    // MARK: - Init

    init() {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 2
        config.timeoutIntervalForResource = 5
        config.waitsForConnectivity = false
        self.session = URLSession(configuration: config)
    }

    deinit {
        disconnect()
    }

    // MARK: - Connection

    /// Attempt to connect to the UGV at the configured host:port.
    /// Sends a test command to verify the device responds.
    func connect() {
        let url = baseURL
        guard let requestURL = URL(string: "\(url)") else { return }

        // Probe: send a no-op status request
        var request = URLRequest(url: requestURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = "{\"T\":130}".data(using: .utf8)

        session.dataTask(with: request) { [weak self] data, response, error in
            guard let self else { return }
            DispatchQueue.main.async {
                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode == 200,
                   data != nil {
                    self.isConnected = true
                    self.startPolling()
                    self.startHeartbeat()
                    self.startRateTracking()
                } else {
                    self.isConnected = false
                }
            }
        }.resume()
    }

    func disconnect() {
        stopPolling()
        stopHeartbeat()
        stopRateTracking()
        stopLogging()
        DispatchQueue.main.async {
            self.isConnected = false
        }
    }

    private var baseURL: String {
        "http://\(hostAddress):\(port)"
    }

    // MARK: - Telemetry Polling

    private func startPolling() {
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.pollTelemetry()
        }
    }

    private func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    private func pollTelemetry() {
        // Poll IMU and chassis in parallel
        sendCommand(["T": 126]) { [weak self] imuData in
            self?.handleIMUResponse(imuData)
        }
        sendCommand(["T": 130]) { [weak self] chassisData in
            self?.handleChassisResponse(chassisData)
        }
    }

    private func sendCommand(_ command: [String: Any], completion: @escaping ([String: Any]?) -> Void) {
        guard let url = URL(string: baseURL) else {
            completion(nil)
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        guard let body = try? JSONSerialization.data(withJSONObject: command) else {
            completion(nil)
            return
        }
        request.httpBody = body

        session.dataTask(with: request) { data, response, error in
            guard let data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                // Connection lost
                if error != nil {
                    DispatchQueue.main.async { [weak self] in
                        self?.isConnected = false
                        self?.stopPolling()
                        self?.stopHeartbeat()
                    }
                }
                completion(nil)
                return
            }
            completion(json)
        }.resume()
    }

    // MARK: - Response Parsing

    private func handleIMUResponse(_ data: [String: Any]?) {
        guard let data else { return }
        let t = clock.nowNanos

        // Waveshare IMU response fields (from QMI8658C + AK09918):
        // heading, accelX, accelY, accelZ, pitch, roll, temperature
        let heading = (data["heading"] as? NSNumber)?.doubleValue ?? 0
        let accelX = (data["accelX"] as? NSNumber)?.doubleValue
            ?? (data["ax"] as? NSNumber)?.doubleValue ?? 0
        let accelY = (data["accelY"] as? NSNumber)?.doubleValue
            ?? (data["ay"] as? NSNumber)?.doubleValue ?? 0
        let accelZ = (data["accelZ"] as? NSNumber)?.doubleValue
            ?? (data["az"] as? NSNumber)?.doubleValue ?? 0
        let pitch = (data["pitch"] as? NSNumber)?.doubleValue ?? 0
        let roll = (data["roll"] as? NSNumber)?.doubleValue ?? 0

        let line = "\(t),imu,\(heading),\(accelX),\(accelY),\(accelZ),\(pitch),\(roll),0,0,0,0\n"
        appendLine(line)
    }

    private func handleChassisResponse(_ data: [String: Any]?) {
        guard let data else { return }
        let t = clock.nowNanos

        // Waveshare chassis response: left/right encoder speeds, battery voltage/current
        let leftSpeed = (data["leftSpeed"] as? NSNumber)?.doubleValue
            ?? (data["L"] as? NSNumber)?.doubleValue ?? 0
        let rightSpeed = (data["rightSpeed"] as? NSNumber)?.doubleValue
            ?? (data["R"] as? NSNumber)?.doubleValue ?? 0
        let voltage = (data["voltage"] as? NSNumber)?.doubleValue
            ?? (data["v"] as? NSNumber)?.doubleValue ?? 0
        let current = (data["current"] as? NSNumber)?.doubleValue
            ?? (data["a"] as? NSNumber)?.doubleValue ?? 0

        DispatchQueue.main.async {
            self.batteryVoltage = voltage
        }

        let line = "\(t),chassis,0,0,0,0,0,0,\(leftSpeed),\(rightSpeed),\(voltage),\(current)\n"
        appendLine(line)
    }

    // MARK: - Heartbeat

    /// Sends a keepalive command every 2 seconds to prevent UGV auto-stop.
    /// Waveshare UGVs stop motors if no command received within 3 seconds.
    private func startHeartbeat() {
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            // Send a zero-speed command as heartbeat (doesn't move the robot)
            self?.sendCommand(["T": 1, "L": 0, "R": 0]) { _ in }
        }
    }

    private func stopHeartbeat() {
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
    }

    // MARK: - Logging

    func setOutputDirectory(_ dir: URL) {
        let url = dir.appendingPathComponent("ugv_telemetry.csv")
        let header = "timestamp_ns,type,heading,accel_x,accel_y,accel_z,pitch,roll,left_speed,right_speed,battery_v,battery_a\n"
        FileManager.default.createFile(atPath: url.path, contents: header.data(using: .utf8))
        outputHandle = try? FileHandle(forWritingTo: url)
        outputHandle?.seekToEndOfFile()
        isLogging = true
        messageCount = 0
    }

    func stopLogging() {
        isLogging = false
        try? outputHandle?.close()
        outputHandle = nil
    }

    private func appendLine(_ line: String) {
        messagesThisSecond += 1
        DispatchQueue.main.async { self.messageCount += 1 }

        guard isLogging, let handle = outputHandle, let data = line.data(using: .utf8) else { return }
        writeQueue.async {
            handle.write(data)
        }
    }

    // MARK: - Rate Tracking

    private func startRateTracking() {
        messagesThisSecond = 0
        rateTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            let count = self.messagesThisSecond
            self.messagesThisSecond = 0
            DispatchQueue.main.async {
                self.telemetryRate = Double(count)
            }
        }
    }

    private func stopRateTracking() {
        rateTimer?.invalidate()
        rateTimer = nil
        telemetryRate = 0
    }
}
