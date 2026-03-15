import Foundation
import Combine

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
    @Published var connectionError: String?

    // MARK: - Private

    private var pollTimer: Timer?
    private var heartbeatTimer: Timer?
    private var outputHandle: FileHandle?
    private let outputLock = NSLock()  // protects isLogging + outputHandle
    private let clock = SyncClock.shared
    private let writeQueue = DispatchQueue(label: "com.sensorforge.wifi.write", qos: .userInitiated)
    private let session: URLSession
    private var messageCounter: Int32 = 0  // atomic counter for thread-safe rate tracking
    private var rateTimer: Timer?
    private var pollIMUNext = true  // alternates IMU/chassis to halve request rate

    // MARK: - Init

    init() {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 2
        config.timeoutIntervalForResource = 5
        config.waitsForConnectivity = false
        self.session = URLSession(configuration: config)
    }

    // MARK: - Connection

    /// Attempt to connect to the UGV at the configured host:port.
    /// Sends a test command to verify the device responds.
    func connect() {
        connectionError = nil
        guard let requestURL = URL(string: baseURL) else {
            connectionError = "Invalid IP address"
            return
        }

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
                    self.connectionError = nil
                    self.startPolling()
                    self.startHeartbeat()
                    self.startRateTracking()
                } else {
                    self.isConnected = false
                    if let error {
                        self.connectionError = error.localizedDescription
                    } else {
                        self.connectionError = "No response from UGV"
                    }
                }
            }
        }.resume()
    }

    func disconnect() {
        stopPolling()
        stopHeartbeat()
        stopRateTracking()
        stopLogging()
        DispatchQueue.main.async { [weak self] in
            self?.isConnected = false
            self?.connectionError = nil
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

    /// Alternates between IMU and chassis requests each tick.
    /// This keeps total rate at 10 req/s instead of 20, avoiding ESP32 overload.
    private func pollTelemetry() {
        if pollIMUNext {
            sendCommand(["T": 126]) { [weak self] imuData in
                self?.handleIMUResponse(imuData)
            }
        } else {
            sendCommand(["T": 130]) { [weak self] chassisData in
                self?.handleChassisResponse(chassisData)
            }
        }
        pollIMUNext.toggle()
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

        let leftSpeed = (data["leftSpeed"] as? NSNumber)?.doubleValue
            ?? (data["L"] as? NSNumber)?.doubleValue ?? 0
        let rightSpeed = (data["rightSpeed"] as? NSNumber)?.doubleValue
            ?? (data["R"] as? NSNumber)?.doubleValue ?? 0
        let voltage = (data["voltage"] as? NSNumber)?.doubleValue
            ?? (data["v"] as? NSNumber)?.doubleValue ?? 0
        let current = (data["current"] as? NSNumber)?.doubleValue
            ?? (data["a"] as? NSNumber)?.doubleValue ?? 0

        DispatchQueue.main.async { [weak self] in
            self?.batteryVoltage = voltage
        }

        let line = "\(t),chassis,0,0,0,0,0,0,\(leftSpeed),\(rightSpeed),\(voltage),\(current)\n"
        appendLine(line)
    }

    // MARK: - Heartbeat

    /// Sends a keepalive command every 2 seconds to prevent UGV auto-stop.
    /// Waveshare UGVs stop motors if no command received within 3 seconds.
    private func startHeartbeat() {
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
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

        outputLock.lock()
        outputHandle = try? FileHandle(forWritingTo: url)
        outputHandle?.seekToEndOfFile()
        isLogging = true
        outputLock.unlock()

        messageCount = 0
    }

    func stopLogging() {
        outputLock.lock()
        isLogging = false
        try? outputHandle?.close()
        outputHandle = nil
        outputLock.unlock()
    }

    /// Thread-safe append. Called from URLSession background threads.
    private func appendLine(_ line: String) {
        OSAtomicIncrement32(&messageCounter)
        DispatchQueue.main.async { [weak self] in
            self?.messageCount += 1
        }

        outputLock.lock()
        guard isLogging, let handle = outputHandle, let data = line.data(using: .utf8) else {
            outputLock.unlock()
            return
        }
        outputLock.unlock()

        writeQueue.async {
            handle.write(data)
        }
    }

    // MARK: - Rate Tracking

    private func startRateTracking() {
        OSAtomicCompareAndSwap32(messageCounter, 0, &messageCounter)
        rateTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            let count = self.messageCounter
            OSAtomicCompareAndSwap32(count, 0, &self.messageCounter)
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
