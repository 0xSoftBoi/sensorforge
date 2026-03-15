import Foundation
import CoreBluetooth
import Combine

/// Generic BLE bridge that connects to ANY Bluetooth device, subscribes to all
/// notify characteristics, and logs raw bytes with synchronized timestamps.
/// Protocol-agnostic by design — decode in post-processing.
final class BLEBridge: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var isScanning = false
    @Published var discoveredDevices: [BLEDevice] = []
    @Published var connectedDevice: BLEDevice?
    @Published var isLogging = false
    @Published var messageCount: Int = 0

    // MARK: - Types

    struct BLEDevice: Identifiable, Hashable {
        let id: UUID
        let name: String
        let peripheral: CBPeripheral

        func hash(into hasher: inout Hasher) { hasher.combine(id) }
        static func == (lhs: BLEDevice, rhs: BLEDevice) -> Bool { lhs.id == rhs.id }
    }

    // MARK: - Private

    private var centralManager: CBCentralManager!
    private var outputHandle: FileHandle?
    private var outputDir: URL?
    private let clock = SyncClock.shared
    private let writeQueue = DispatchQueue(label: "com.sensorforge.ble.write", qos: .userInitiated)

    // MARK: - Init

    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }

    // MARK: - Scanning

    func startScan() {
        guard centralManager.state == .poweredOn else { return }
        discoveredDevices.removeAll()
        isScanning = true
        centralManager.scanForPeripherals(withServices: nil, options: [
            CBCentralManagerScanOptionAllowDuplicatesKey: false
        ])

        // Auto-stop scan after 10 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) { [weak self] in
            self?.stopScan()
        }
    }

    func stopScan() {
        centralManager.stopScan()
        isScanning = false
    }

    // MARK: - Connection

    func connect(to device: BLEDevice) {
        stopScan()
        centralManager.connect(device.peripheral, options: nil)
    }

    func disconnect() {
        if let peripheral = connectedDevice?.peripheral {
            centralManager.cancelPeripheralConnection(peripheral)
        }
        connectedDevice = nil
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
    }

    func stopLogging() {
        isLogging = false
        try? outputHandle?.close()
        outputHandle = nil
    }

    private func logValue(serviceUUID: String, charUUID: String, data: Data) {
        guard isLogging, let handle = outputHandle else { return }
        let t = clock.nowNanos
        let hex = data.map { String(format: "%02x", $0) }.joined()
        let name = connectedDevice?.name ?? "unknown"
        let line = "\(t),\(name),\(serviceUUID),\(charUUID),\(hex)\n"

        if let lineData = line.data(using: .utf8) {
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
            print("BLE: Powered on")
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any], rssi RSSI: NSNumber) {
        let name = peripheral.name ?? advertisementData[CBAdvertisementDataLocalNameKey] as? String ?? "Unknown"
        // Skip unnamed devices to reduce noise
        guard name != "Unknown" else { return }

        let device = BLEDevice(id: peripheral.identifier, name: name, peripheral: peripheral)
        if !discoveredDevices.contains(where: { $0.id == device.id }) {
            DispatchQueue.main.async {
                self.discoveredDevices.append(device)
            }
        }
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        let name = peripheral.name ?? "Device"
        connectedDevice = BLEDevice(id: peripheral.identifier, name: name, peripheral: peripheral)
        peripheral.delegate = self
        peripheral.discoverServices(nil)
        print("BLE: Connected to \(name)")
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        print("BLE: Disconnected")
        DispatchQueue.main.async { self.connectedDevice = nil }
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        print("BLE: Connection failed: \(error?.localizedDescription ?? "unknown")")
    }
}

// MARK: - CBPeripheralDelegate

extension BLEBridge: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        for service in services {
            peripheral.discoverCharacteristics(nil, for: service)
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let chars = service.characteristics else { return }
        for char in chars {
            // Subscribe to all notify/indicate characteristics
            if char.properties.contains(.notify) || char.properties.contains(.indicate) {
                peripheral.setNotifyValue(true, for: char)
            }
            // Read any readable characteristics once
            if char.properties.contains(.read) {
                peripheral.readValue(for: char)
            }
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let data = characteristic.value else { return }
        let serviceUUID = characteristic.service?.uuid.uuidString ?? "unknown"
        let charUUID = characteristic.uuid.uuidString
        logValue(serviceUUID: serviceUUID, charUUID: charUUID, data: data)
    }
}
