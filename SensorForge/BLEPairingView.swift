import SwiftUI

/// Full-screen BLE pairing experience — scan, see signal strength, tap to pair,
/// auto-reconnect to saved devices. Feels like pairing a JBL speaker.
struct BLEPairingView: View {
    @ObservedObject var ble: BLEBridge
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            ZStack {
                Color.black.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 24) {
                        // Connected device card
                        if let device = ble.connectedDevice {
                            connectedCard(device)
                        }

                        // Connection state banner
                        connectionBanner

                        // Saved devices
                        if !ble.savedDevices.isEmpty && ble.connectedDevice == nil {
                            savedSection
                        }

                        // Discovered devices
                        if !ble.discoveredDevices.isEmpty {
                            discoveredSection
                        }

                        // Empty state
                        if ble.discoveredDevices.isEmpty && ble.connectedDevice == nil {
                            emptyState
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Devices")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") { dismiss() }
                        .foregroundColor(.white)
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    scanButton
                }
            }
            .onAppear {
                if ble.connectedDevice == nil {
                    ble.startScan()
                }
            }
            .onDisappear {
                ble.stopScan()
            }
        }
    }

    // MARK: - Connected Card

    private func connectedCard(_ device: BLEDevice) -> some View {
        VStack(spacing: 16) {
            // Device icon with pulse
            ZStack {
                // Outer pulse rings
                Circle()
                    .stroke(deviceColor(device.type).opacity(0.15), lineWidth: 2)
                    .frame(width: 120, height: 120)
                Circle()
                    .stroke(deviceColor(device.type).opacity(0.3), lineWidth: 2)
                    .frame(width: 90, height: 90)

                Circle()
                    .fill(deviceColor(device.type).opacity(0.15))
                    .frame(width: 72, height: 72)
                Image(systemName: device.type.icon)
                    .font(.system(size: 28))
                    .foregroundColor(deviceColor(device.type))
            }

            VStack(spacing: 4) {
                Text(device.name)
                    .font(.title2.bold())
                    .foregroundColor(.white)
                Text("Connected")
                    .font(.subheadline)
                    .foregroundColor(.green)
            }

            // Stats row
            HStack(spacing: 24) {
                statPill(icon: "wifi", value: "\(device.signalBars)/4", label: "Signal")
                statPill(icon: "number", value: "\(device.characteristicCount)", label: "Channels")
                if ble.isLogging {
                    statPill(icon: "arrow.down.circle", value: ble.dataRate, label: "Data")
                }
            }

            // Disconnect button
            Button {
                ble.disconnect()
            } label: {
                Text("Disconnect")
                    .font(.subheadline.bold())
                    .foregroundColor(.red)
                    .frame(maxWidth: .infinity)
                    .frame(height: 44)
                    .background(Color.red.opacity(0.15))
                    .cornerRadius(12)
            }
        }
        .padding(20)
        .background(Color.white.opacity(0.05))
        .cornerRadius(16)
    }

    private func statPill(icon: String, value: String, label: String) -> some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption2)
                Text(value)
                    .font(.caption.bold().monospaced())
            }
            .foregroundColor(.white)
            Text(label)
                .font(.caption2)
                .foregroundColor(.gray)
        }
    }

    // MARK: - Connection Banner

    @ViewBuilder
    private var connectionBanner: some View {
        switch ble.connectionState {
        case .scanning:
            HStack(spacing: 8) {
                ProgressView()
                    .tint(.blue)
                Text("Searching for devices...")
                    .font(.subheadline)
                    .foregroundColor(.gray)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)

        case .connecting(let name):
            HStack(spacing: 8) {
                ProgressView()
                    .tint(.orange)
                Text("Connecting to \(name)...")
                    .font(.subheadline)
                    .foregroundColor(.orange)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(Color.orange.opacity(0.1))
            .cornerRadius(12)

        case .discovering(let name):
            HStack(spacing: 8) {
                ProgressView()
                    .tint(.blue)
                Text("Setting up \(name)...")
                    .font(.subheadline)
                    .foregroundColor(.blue)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(Color.blue.opacity(0.1))
            .cornerRadius(12)

        case .reconnecting(let name):
            HStack(spacing: 8) {
                ProgressView()
                    .tint(.purple)
                Text("Reconnecting to \(name)...")
                    .font(.subheadline)
                    .foregroundColor(.purple)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(Color.purple.opacity(0.1))
            .cornerRadius(12)

        case .connected, .disconnected:
            EmptyView()
        }
    }

    // MARK: - Saved Devices

    private var savedSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("MY DEVICES")
                .font(.caption.bold())
                .foregroundColor(.gray)
                .padding(.leading, 4)

            ForEach(ble.savedDevices) { saved in
                savedDeviceRow(saved)
            }
        }
    }

    private func savedDeviceRow(_ saved: SavedBLEDevice) -> some View {
        Button {
            ble.connectToSaved(saved)
        } label: {
            HStack(spacing: 12) {
                // Icon
                ZStack {
                    Circle()
                        .fill(deviceColor(saved.type).opacity(0.15))
                        .frame(width: 44, height: 44)
                    Image(systemName: saved.type.icon)
                        .font(.body)
                        .foregroundColor(deviceColor(saved.type))
                }

                // Name + info
                VStack(alignment: .leading, spacing: 2) {
                    Text(saved.name)
                        .font(.body.bold())
                        .foregroundColor(.white)
                    Text("Last seen \(relativeDate(saved.lastSeenDate))")
                        .font(.caption)
                        .foregroundColor(.gray)
                }

                Spacer()

                Image(systemName: "arrow.right.circle")
                    .foregroundColor(.gray)
            }
            .padding(12)
            .background(Color.white.opacity(0.05))
            .cornerRadius(12)
        }
        .contextMenu {
            Button(role: .destructive) {
                ble.forgetDevice(saved.id)
            } label: {
                Label("Forget Device", systemImage: "trash")
            }
        }
    }

    // MARK: - Discovered Devices

    private var discoveredSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("NEARBY")
                .font(.caption.bold())
                .foregroundColor(.gray)
                .padding(.leading, 4)

            ForEach(ble.discoveredDevices) { device in
                // Don't show already-connected device in the list
                if device.id != ble.connectedDevice?.id {
                    discoveredDeviceRow(device)
                }
            }
        }
    }

    private func discoveredDeviceRow(_ device: BLEDevice) -> some View {
        Button {
            ble.connect(to: device)
        } label: {
            HStack(spacing: 12) {
                // Icon
                ZStack {
                    Circle()
                        .fill(deviceColor(device.type).opacity(0.15))
                        .frame(width: 44, height: 44)
                    Image(systemName: device.type.icon)
                        .font(.body)
                        .foregroundColor(deviceColor(device.type))
                }

                // Name + type
                VStack(alignment: .leading, spacing: 2) {
                    Text(device.name)
                        .font(.body.bold())
                        .foregroundColor(.white)
                    Text(device.type.rawValue)
                        .font(.caption)
                        .foregroundColor(.gray)
                }

                Spacer()

                // Signal bars
                signalIndicator(bars: device.signalBars)
            }
            .padding(12)
            .background(Color.white.opacity(0.05))
            .cornerRadius(12)
        }
    }

    // MARK: - Signal Indicator

    private func signalIndicator(bars: Int) -> some View {
        HStack(spacing: 2) {
            ForEach(1...4, id: \.self) { bar in
                RoundedRectangle(cornerRadius: 1)
                    .fill(bar <= bars ? signalColor(bars) : Color.gray.opacity(0.3))
                    .frame(width: 4, height: CGFloat(bar) * 4 + 4)
            }
        }
        .frame(height: 20, alignment: .bottom)
    }

    private func signalColor(_ bars: Int) -> Color {
        switch bars {
        case 4: return .green
        case 3: return .green
        case 2: return .yellow
        default: return .red
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "antenna.radiowaves.left.and.right")
                .font(.system(size: 48))
                .foregroundColor(.gray.opacity(0.5))

            if ble.isScanning {
                Text("Looking for nearby devices...")
                    .font(.body)
                    .foregroundColor(.gray)
                Text("Make sure your device is powered on\nand Bluetooth is enabled")
                    .font(.caption)
                    .foregroundColor(.gray.opacity(0.7))
                    .multilineTextAlignment(.center)
            } else {
                Text("No devices found")
                    .font(.body)
                    .foregroundColor(.gray)
                Button("Scan Again") {
                    ble.startScan()
                }
                .foregroundColor(.blue)
            }
        }
        .padding(.top, 48)
    }

    // MARK: - Scan Button

    private var scanButton: some View {
        Button {
            if ble.isScanning {
                ble.stopScan()
            } else {
                ble.startScan()
            }
        } label: {
            if ble.isScanning {
                ProgressView()
                    .tint(.blue)
            } else {
                Image(systemName: "arrow.clockwise")
                    .foregroundColor(.blue)
            }
        }
    }

    // MARK: - Helpers

    private func deviceColor(_ type: BLEDeviceType) -> Color {
        switch type {
        case .traxxas:    return .red
        case .arduino:    return .teal
        case .esp32:      return .blue
        case .obdII:      return .orange
        case .raspberryPi: return .green
        case .generic:    return .gray
        }
    }

    private func relativeDate(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}
