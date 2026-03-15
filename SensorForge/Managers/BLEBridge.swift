import Foundation
import CoreBluetooth
import Combine

/// Discovers and connects to BLE peripherals, logging their telemetry to the shared data store.
///
/// Supports generic GATT services. Device-specific parsers (Traxxas, OBD-II, etc.)
/// can be registered via `registerParser(_:for:)`.
final class BLEBridge: NSObject, ObservableObject {
    @Published var isScanning = false
    @Published var discoveredDevices: [BLEDevice] = []
    @Published var connectedDevices: [BLEDevice] = []

    private var centralManager: CBCentralManager?
    private var dataStore: SensorDataStore?
    private var peripherals: [UUID: CBPeripheral] = [:]
    private var parsers: [String: BLETelemetryParser] = [:]

    struct BLEDevice: Identifiable {
        let id: UUID
        let name: String
        var rssi: Int
        var isConnected: Bool = false
        var services: [String] = []
    }

    override init() {
        super.init()
    }

    func start(dataStore: SensorDataStore) {
        self.dataStore = dataStore
        if centralManager == nil {
            centralManager = CBCentralManager(delegate: self, queue: .main)
        }
    }

    func stop() {
        centralManager?.stopScan()
        for peripheral in peripherals.values {
            centralManager?.cancelPeripheralConnection(peripheral)
        }
        peripherals.removeAll()
        discoveredDevices.removeAll()
        connectedDevices.removeAll()
        isScanning = false
    }

    func startScanning() {
        guard centralManager?.state == .poweredOn else { return }
        centralManager?.scanForPeripherals(withServices: nil, options: [
            CBCentralManagerScanOptionAllowDuplicatesKey: false
        ])
        isScanning = true
    }

    func stopScanning() {
        centralManager?.stopScan()
        isScanning = false
    }

    func connect(to device: BLEDevice) {
        guard let peripheral = peripherals[device.id] else { return }
        centralManager?.connect(peripheral, options: nil)
    }

    func disconnect(from device: BLEDevice) {
        guard let peripheral = peripherals[device.id] else { return }
        centralManager?.cancelPeripheralConnection(peripheral)
    }

    /// Register a custom parser for a specific service UUID.
    func registerParser(_ parser: BLETelemetryParser, for serviceUUID: String) {
        parsers[serviceUUID] = parser
    }
}

// MARK: - Telemetry Parser Protocol

protocol BLETelemetryParser {
    func parse(data: Data, characteristicUUID: String) -> [String: String]
}

/// Default parser that just converts bytes to hex.
struct GenericBLEParser: BLETelemetryParser {
    func parse(data: Data, characteristicUUID: String) -> [String: String] {
        ["raw_hex": data.map { String(format: "%02x", $0) }.joined(separator: " ")]
    }
}

// MARK: - CBCentralManagerDelegate

extension BLEBridge: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        if central.state == .poweredOn && isScanning {
            startScanning()
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                         advertisementData: [String: Any], rssi RSSI: NSNumber) {
        let name = peripheral.name ?? advertisementData[CBAdvertisementDataLocalNameKey] as? String ?? "Unknown"
        let id = peripheral.identifier

        peripherals[id] = peripheral

        if let index = discoveredDevices.firstIndex(where: { $0.id == id }) {
            discoveredDevices[index].rssi = RSSI.intValue
        } else {
            discoveredDevices.append(BLEDevice(id: id, name: name, rssi: RSSI.intValue))
        }
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        peripheral.delegate = self
        peripheral.discoverServices(nil)

        if let index = discoveredDevices.firstIndex(where: { $0.id == peripheral.identifier }) {
            discoveredDevices[index].isConnected = true
            // Avoid duplicate entries in connectedDevices
            if !connectedDevices.contains(where: { $0.id == peripheral.identifier }) {
                connectedDevices.append(discoveredDevices[index])
            }
        }
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        connectedDevices.removeAll { $0.id == peripheral.identifier }
        if let index = discoveredDevices.firstIndex(where: { $0.id == peripheral.identifier }) {
            discoveredDevices[index].isConnected = false
        }
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
        guard let characteristics = service.characteristics else { return }
        for characteristic in characteristics {
            if characteristic.properties.contains(.notify) {
                peripheral.setNotifyValue(true, for: characteristic)
            }
            if characteristic.properties.contains(.read) {
                peripheral.readValue(for: characteristic)
            }
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        if let error {
            print("[BLEBridge] Characteristic read error: \(error)")
            return
        }
        guard let value = characteristic.value else { return }

        let ts = TimestampProvider.shared.now
        let serviceUUID = characteristic.service?.uuid.uuidString ?? ""
        let parser = parsers[serviceUUID] ?? GenericBLEParser()
        let parsed = parser.parse(data: value, characteristicUUID: characteristic.uuid.uuidString)

        let telemetry = BLETelemetry(
            timestamp: ts,
            deviceName: peripheral.name ?? "Unknown",
            deviceUUID: peripheral.identifier.uuidString,
            characteristicUUID: characteristic.uuid.uuidString,
            rawHex: value.map { String(format: "%02x", $0) }.joined(separator: " "),
            parsedValues: parsed
        )

        Task { @MainActor in
            self.dataStore?.bleTelemetry.append(telemetry)
        }
    }
}
