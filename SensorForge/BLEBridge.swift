import Foundation
import CoreBluetooth
import Combine

// MARK: - Device Profile

/// Recognized device types based on advertised name patterns or service UUIDs.
enum BLEDeviceType: String, Codable, CaseIterable {
    case traxxas = "Traxxas"
    case arduino = "Arduino"
    case esp32 = "ESP32"
    case obdII = "OBD-II"
    case raspberryPi = "RPi"
    case generic = "Device"

    var icon: String {
        switch self {
        case .traxxas:    return "car.fill"
        case .arduino:    return "cpu"
        case .esp32:      return "cpu"
        case .obdII:      return "car.fill"
        case .raspberryPi: return "desktopcomputer"
        case .generic:    return "antenna.radiowaves.left.and.right"
        }
    }

    var color: String {
        switch self {
        case .traxxas:    return "red"
        case .arduino:    return "teal"
        case .esp32:      return "blue"
        case .obdII:      return "orange"
        case .raspberryPi: return "green"
        case .generic:    return "gray"
        }
    }

    static func detect(name: String, services: [CBUUID]?) -> BLEDeviceType {
        let lower = name.lowercased()
        if lower.contains("traxxas") || lower.contains("tqi") { return .traxxas }
        if lower.contains("arduino") || lower.contains("hm-10") || lower.contains("bluno") { return .arduino }
        if lower.contains("esp") { return .esp32 }
        if lower.contains("obd") || lower.contains("elm") || lower.contains("vlink") || lower.contains("veepeak") { return .obdII }
        if lower.contains("raspberry") || lower.contains("rpi") { return .raspberryPi }
        return .generic
    }
}

// MARK: - BLE Device

struct BLEDevice: Identifiable, Hashable {
    let id: UUID
    let name: String
    let type: BLEDeviceType
    var rssi: Int
    var peripheral: CBPeripheral?
    var characteristicCount: Int = 0

    /// Signal quality: 0.0 (terrible) to 1.0 (great)
    var signalQuality: Double {
        // RSSI ranges: -30 dBm (amazing) to -100 dBm (unusable)
        let clamped = min(max(Double(rssi), -100), -30)
        return (clamped + 100) / 70.0
    }

    var signalBars: Int {
        switch rssi {
        case -50...0:    return 4
        case -65...(-51): return 3
        case -80...(-66): return 2
        default:          return 1
        }
    }

    func hash(into hasher: inout Hasher) { hasher.combine(id) }
    static func == (lhs: BLEDevice, rhs: BLEDevice) -> Bool { lhs.id == rhs.id }
}

// MARK: - Saved Device (for persistence)

struct SavedBLEDevice: Codable, Identifiable {
    let id: UUID  // CBPeripheral identifier
    let name: String
    let type: BLEDeviceType
    let pairedDate: Date
    var lastSeenDate: Date
}

// MARK: - Connection State

enum BLEConnectionState: Equatable {
    case disconnected
    case scanning
    case connecting(deviceName: String)
    case discovering(deviceName: String)
    case connected(deviceName: String)
    case reconnecting(deviceName: String)
}

// MARK: - BLE Bridge

/// Consumer-grade BLE bridge: scan, pair like a JBL speaker, auto-reconnect,
/// remember devices, subscribe to all GATT characteristics, log raw bytes.
final class BLEBridge: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var connectionState: BLEConnectionState = .disconnected
    @Published var discoveredDevices: [BLEDevice] = []
    @Published var savedDevices: [SavedBLEDevice] = []
    @Published var connectedDevice: BLEDevice?
    @Published var isLogging = false
    @Published var messageCount: Int = 0
    @Published var dataRate: String = "0 B/s"

    var isScanning: Bool {
        if case .scanning = connectionState { return true }
        return false
    }

    // MARK: - Private

    private var centralManager: CBCentralManager!
    private var peripheralMap: [UUID: CBPeripheral] = [:]  // retain peripherals
    private var outputHandle: FileHandle?
    private var outputDir: URL?
    private let clock = SyncClock.shared
    private let writeQueue = DispatchQueue(label: "com.sensorforge.ble.write", qos: .userInitiated)
    private var scanTimer: Timer?
    private var rssiTimer: Timer?
    private var bytesThisSecond: Int = 0
    private var dataRateTimer: Timer?
    private var autoReconnectID: UUID?

    private static let savedDevicesKey = "com.sensorforge.ble.savedDevices"

    // MARK: - Init

    override init() {
        super.init()
        loadSavedDevices()
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }

    // MARK: - Persistence

    private func loadSavedDevices() {
        guard let data = UserDefaults.standard.data(forKey: Self.savedDevicesKey),
              let devices = try? JSONDecoder().decode([SavedBLEDevice].self, from: data) else { return }
        savedDevices = devices
    }

    private func saveToDisk() {
        if let data = try? JSONEncoder().encode(savedDevices) {
            UserDefaults.standard.set(data, forKey: Self.savedDevicesKey)
        }
    }

    private func rememberDevice(_ device: BLEDevice) {
        if let idx = savedDevices.firstIndex(where: { $0.id == device.id }) {
            savedDevices[idx].lastSeenDate = Date()
        } else {
            let saved = SavedBLEDevice(id: device.id, name: device.name, type: device.type,
                                        pairedDate: Date(), lastSeenDate: Date())
            savedDevices.append(saved)
        }
        saveToDisk()
    }

    func forgetDevice(_ deviceID: UUID) {
        savedDevices.removeAll { $0.id == deviceID }
        saveToDisk()
        if connectedDevice?.id == deviceID {
            disconnect()
        }
    }

    // MARK: - Scanning

    func startScan() {
        guard centralManager.state == .poweredOn else { return }
        discoveredDevices.removeAll()
        connectionState = .scanning

        centralManager.scanForPeripherals(withServices: nil, options: [
            CBCentralManagerScanOptionAllowDuplicatesKey: false
        ])

        // Auto-stop scan after 15 seconds
        scanTimer?.invalidate()
        scanTimer = Timer.scheduledTimer(withTimeInterval: 15, repeats: false) { [weak self] _ in
            self?.stopScan()
        }
    }

    func stopScan() {
        centralManager.stopScan()
        scanTimer?.invalidate()
        scanTimer = nil
        if case .scanning = connectionState {
            connectionState = .disconnected
        }
    }

    // MARK: - Connection

    func connect(to device: BLEDevice) {
        guard let peripheral = device.peripheral ?? peripheralMap[device.id] else { return }
        stopScan()
        connectionState = .connecting(deviceName: device.name)
        centralManager.connect(peripheral, options: nil)
    }

    func connectToSaved(_ saved: SavedBLEDevice) {
        // Try to find the peripheral by known identifier
        let peripherals = centralManager.retrievePeripherals(withIdentifiers: [saved.id])
        if let peripheral = peripherals.first {
            peripheralMap[saved.id] = peripheral
            connectionState = .reconnecting(deviceName: saved.name)
            centralManager.connect(peripheral, options: nil)
        } else {
            // Peripheral not cached — need to scan for it
            autoReconnectID = saved.id
            startScan()
        }
    }

    func disconnect() {
        if let peripheral = connectedDevice?.peripheral {
            centralManager.cancelPeripheralConnection(peripheral)
        }
        connectedDevice = nil
        connectionState = .disconnected
        stopRSSIPolling()
        stopDataRateTracking()
    }

    /// Called on app launch — tries to reconnect to the most recently used device.
    func autoReconnect() {
        guard let mostRecent = savedDevices.sorted(by: { $0.lastSeenDate > $1.lastSeenDate }).first else { return }
        connectToSaved(mostRecent)
    }

    // MARK: - RSSI Polling

    private func startRSSIPolling() {
        rssiTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.connectedDevice?.peripheral?.readRSSI()
        }
    }

    private func stopRSSIPolling() {
        rssiTimer?.invalidate()
        rssiTimer = nil
    }

    // MARK: - Data Rate Tracking

    private func startDataRateTracking() {
        bytesThisSecond = 0
        dataRateTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            let bytes = self.bytesThisSecond
            self.bytesThisSecond = 0
            DispatchQueue.main.async {
                if bytes > 1024 {
                    self.dataRate = String(format: "%.1f KB/s", Double(bytes) / 1024)
                } else {
                    self.dataRate = "\(bytes) B/s"
                }
            }
        }
    }

    private func stopDataRateTracking() {
        dataRateTimer?.invalidate()
        dataRateTimer = nil
        dataRate = "0 B/s"
    }

    // MARK: - Logging

    func setOutputDirectory(_ dir: URL) {
        outputDir = dir
        let url = dir.appendingPathComponent("ble_telemetry.csv")
        let header = "timestamp_ns,device_name,service_uuid,char_uuid,hex_value\n"
        FileManager.default.createFile(atPath: url.path, contents: header.data(using: .utf8))
        outputHandle = try? FileHandle(forWritingTo: url)
        outputHandle?.seekToEndOfFile()
        isLogging = true
        messageCount = 0
        startDataRateTracking()
    }

    func stopLogging() {
        isLogging = false
        try? outputHandle?.close()
        outputHandle = nil
        stopDataRateTracking()
    }

    private func logValue(serviceUUID: String, charUUID: String, data: Data) {
        let t = clock.nowNanos
        let hex = data.map { String(format: "%02x", $0) }.joined()
        let name = connectedDevice?.name ?? "unknown"
        let line = "\(t),\(name),\(serviceUUID),\(charUUID),\(hex)\n"

        bytesThisSecond += data.count

        if isLogging, let handle = outputHandle, let lineData = line.data(using: .utf8) {
            writeQueue.async {
                handle.write(lineData)
            }
        }
        DispatchQueue.main.async { self.messageCount += 1 }
    }
}

// MARK: - CBCentralManagerDelegate

extension BLEBridge: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        if central.state == .poweredOn {
            // Auto-reconnect to last device on BLE ready
            if connectedDevice == nil && !savedDevices.isEmpty {
                autoReconnect()
            }
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any], rssi RSSI: NSNumber) {
        let name = peripheral.name
            ?? advertisementData[CBAdvertisementDataLocalNameKey] as? String
            ?? ""
        guard !name.isEmpty else { return }

        let rssiInt = RSSI.intValue
        guard rssiInt != 127 else { return }  // 127 = unavailable

        let serviceUUIDs = advertisementData[CBAdvertisementDataServiceUUIDsKey] as? [CBUUID]
        let type = BLEDeviceType.detect(name: name, services: serviceUUIDs)

        // Retain the peripheral
        peripheralMap[peripheral.identifier] = peripheral

        // Check if this is the device we're auto-reconnecting to
        if let targetID = autoReconnectID, peripheral.identifier == targetID {
            autoReconnectID = nil
            stopScan()
            let device = BLEDevice(id: peripheral.identifier, name: name, type: type, rssi: rssiInt, peripheral: peripheral)
            connectionState = .reconnecting(deviceName: name)
            centralManager.connect(peripheral, options: nil)
            // Also add to discovered list
            if !discoveredDevices.contains(where: { $0.id == device.id }) {
                discoveredDevices.append(device)
            }
            return
        }

        DispatchQueue.main.async {
            if let idx = self.discoveredDevices.firstIndex(where: { $0.id == peripheral.identifier }) {
                // Update RSSI for existing device
                self.discoveredDevices[idx].rssi = rssiInt
            } else {
                let device = BLEDevice(id: peripheral.identifier, name: name, type: type,
                                       rssi: rssiInt, peripheral: peripheral)
                self.discoveredDevices.append(device)
            }
            // Sort by signal strength (closest first)
            self.discoveredDevices.sort { $0.rssi > $1.rssi }
        }
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        let name = peripheral.name ?? "Device"
        let type = discoveredDevices.first(where: { $0.id == peripheral.identifier })?.type ?? .generic
        let rssi = discoveredDevices.first(where: { $0.id == peripheral.identifier })?.rssi ?? -70

        connectionState = .discovering(deviceName: name)
        peripheral.delegate = self
        peripheral.discoverServices(nil)

        let device = BLEDevice(id: peripheral.identifier, name: name, type: type,
                               rssi: rssi, peripheral: peripheral)
        connectedDevice = device
        rememberDevice(device)
        startRSSIPolling()
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        let wasConnected = connectedDevice?.id == peripheral.identifier
        connectedDevice = nil
        connectionState = .disconnected
        stopRSSIPolling()
        stopDataRateTracking()

        // Auto-reconnect if unexpected disconnect
        if wasConnected, error != nil,
           let saved = savedDevices.first(where: { $0.id == peripheral.identifier }) {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
                self?.connectToSaved(saved)
            }
        }
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        connectionState = .disconnected
    }
}

// MARK: - CBPeripheralDelegate

extension BLEBridge: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else {
            // No services — still mark as connected
            connectionState = .connected(deviceName: peripheral.name ?? "Device")
            return
        }
        for service in services {
            peripheral.discoverCharacteristics(nil, for: service)
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let chars = service.characteristics else { return }

        for char in chars {
            if char.properties.contains(.notify) || char.properties.contains(.indicate) {
                peripheral.setNotifyValue(true, for: char)
            }
            if char.properties.contains(.read) {
                peripheral.readValue(for: char)
            }
        }

        if var device = connectedDevice {
            device.characteristicCount += chars.count
            connectedDevice = device
        }
        connectionState = .connected(deviceName: peripheral.name ?? "Device")
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let data = characteristic.value else { return }
        let serviceUUID = characteristic.service?.uuid.uuidString ?? "unknown"
        let charUUID = characteristic.uuid.uuidString
        logValue(serviceUUID: serviceUUID, charUUID: charUUID, data: data)
    }

    func peripheral(_ peripheral: CBPeripheral, didReadRSSI RSSI: NSNumber, error: Error?) {
        guard error == nil else { return }
        DispatchQueue.main.async {
            if var device = self.connectedDevice {
                device.rssi = RSSI.intValue
                self.connectedDevice = device
            }
        }
    }
}
