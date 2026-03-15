import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var coordinator: RecordingCoordinator
    @EnvironmentObject var shakeDetector: ShakeDetector

    @AppStorage("imuRate") private var imuRate = 200
    @AppStorage("enableShakeDetection") private var enableShakeDetection = true
    @AppStorage("defaultCountdown") private var defaultCountdown = 10
    @AppStorage("enableBLEScan") private var enableBLEScan = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Sensor Configuration") {
                    Picker("IMU Rate", selection: $imuRate) {
                        Text("100 Hz").tag(100)
                        Text("200 Hz").tag(200)
                    }

                    Toggle("Auto-scan BLE on launch", isOn: $enableBLEScan)
                }

                Section("Recording Triggers") {
                    Toggle("Shake to Record (3x)", isOn: $enableShakeDetection)
                        .onChange(of: enableShakeDetection) { _, newValue in
                            shakeDetector.isEnabled = newValue
                        }

                    Picker("Default Countdown", selection: $defaultCountdown) {
                        ForEach(CountdownTimer.presets, id: \.self) { seconds in
                            Text("\(seconds)s").tag(seconds)
                        }
                    }
                }

                Section("BLE Devices") {
                    if coordinator.bleBridge.connectedDevices.isEmpty {
                        Button {
                            coordinator.bleBridge.start(dataStore: coordinator.dataStore)
                            coordinator.bleBridge.startScanning()
                        } label: {
                            Label("Scan for Devices", systemImage: "antenna.radiowaves.left.and.right")
                        }
                    }

                    ForEach(coordinator.bleBridge.discoveredDevices) { device in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(device.name)
                                    .font(.body)
                                Text("RSSI: \(device.rssi) dBm")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }

                            Spacer()

                            if device.isConnected {
                                Button("Disconnect") {
                                    coordinator.bleBridge.disconnect(from: device)
                                }
                                .foregroundColor(.red)
                            } else {
                                Button("Connect") {
                                    coordinator.bleBridge.connect(to: device)
                                }
                            }
                        }
                    }
                }

                Section("Export") {
                    HStack {
                        Text("Format")
                        Spacer()
                        Text("CSV")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Coming Soon")
                        Spacer()
                        Text("MCAP, LeRobot, ROS2, HDF5")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Section("Storage") {
                    let sessions = SessionStore.shared.loadSessions()
                    HStack {
                        Text("Sessions")
                        Spacer()
                        Text("\(sessions.count)")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Total Size")
                        Spacer()
                        Text(totalSize(sessions))
                            .foregroundColor(.secondary)
                    }
                }

                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("0.1.0")
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Device")
                        Spacer()
                        Text(UIDevice.current.model)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }

    private func totalSize(_ sessions: [SessionMetadata]) -> String {
        let bytes = sessions.compactMap(\.totalFileSizeBytes).reduce(0, +)
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
