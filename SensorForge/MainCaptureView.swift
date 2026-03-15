import SwiftUI

struct MainCaptureView: View {
    @ObservedObject var engine: CaptureEngine
    @ObservedObject var ble: BLEBridge
    @StateObject private var countdown = CountdownStarter()
    @State private var selectedDelay: Int = 0
    @State private var showBLESheet = false
    @State private var showShareSheet = false

    private let delays = [0, 10, 30, 60]

    var body: some View {
        VStack(spacing: 0) {
            // MARK: - Sensor Status Grid
            sensorGrid
                .padding(.horizontal)
                .padding(.top, 8)

            Spacer()

            // MARK: - Center Display
            if countdown.isCountingDown {
                countdownDisplay
            } else if engine.isRecording {
                recordingStats
            } else {
                idleDisplay
            }

            Spacer()

            // MARK: - Controls
            controlsSection
                .padding(.bottom, 32)
        }
        .background(Color.black)
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showBLESheet) { BLEPairingView(ble: ble) }
        .sheet(isPresented: $showShareSheet) { shareSheet }
        .onReceive(NotificationCenter.default.publisher(for: .siriStartRecording)) { _ in
            if !engine.isRecording { startCapture() }
        }
        .onReceive(NotificationCenter.default.publisher(for: .siriStopRecording)) { _ in
            if engine.isRecording { stopCapture() }
        }
    }

    // MARK: - Sensor Grid

    private var sensorGrid: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 3), spacing: 8) {
            SensorTile(label: "IMU", active: engine.imuActive, icon: "gyroscope")
            SensorTile(label: "GPS", active: engine.gpsActive, icon: "location.fill")
            SensorTile(label: "DEPTH", active: engine.depthActive, icon: "camera.metering.matrix")
            SensorTile(label: "BARO", active: engine.baroActive, icon: "barometer")
            SensorTile(label: "AUDIO", active: engine.audioActive, icon: "waveform")
            SensorTile(label: "BLE", active: ble.connectedDevice != nil, icon: "antenna.radiowaves.left.and.right")
        }
    }

    // MARK: - Countdown

    private var countdownDisplay: some View {
        VStack(spacing: 12) {
            Text("\(countdown.secondsRemaining)")
                .font(.system(size: 120, weight: .thin, design: .monospaced))
                .foregroundColor(.orange)
            Text("Starting soon...")
                .font(.subheadline)
                .foregroundColor(.gray)
            Button("Cancel") {
                countdown.cancel()
            }
            .foregroundColor(.red)
        }
    }

    // MARK: - Recording Stats

    private var recordingStats: some View {
        VStack(spacing: 16) {
            // Recording indicator
            HStack(spacing: 8) {
                Circle()
                    .fill(.red)
                    .frame(width: 12, height: 12)
                    .opacity(pulsingOpacity)
                    .onAppear {
                        withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                            pulsingOpacity = 0.3
                        }
                    }
                    .onDisappear { pulsingOpacity = 1.0 }
                Text("RECORDING")
                    .font(.headline)
                    .foregroundColor(.red)
            }

            // Stats
            VStack(spacing: 8) {
                StatRow(label: "Duration", value: formatDuration(engine.recordingDuration))
                StatRow(label: "Frames", value: "\(engine.frameCount)")
                StatRow(label: "Storage", value: String(format: "%.1f MB", engine.storageUsedMB))
                if ble.isLogging {
                    StatRow(label: "BLE msgs", value: "\(ble.messageCount)")
                }
            }
            .padding()
            .background(Color.white.opacity(0.05))
            .cornerRadius(12)
        }
        .padding(.horizontal, 32)
    }

    @State private var pulsingOpacity: Double = 1.0

    // MARK: - Idle Display

    private var idleDisplay: some View {
        VStack(spacing: 12) {
            Image(systemName: "sensor.fill")
                .font(.system(size: 48))
                .foregroundColor(.gray)
            Text("Ready to capture")
                .font(.title3)
                .foregroundColor(.gray)
            if let device = ble.connectedDevice {
                HStack(spacing: 4) {
                    Image(systemName: device.type.icon)
                    Text(device.name)
                }
                .font(.caption)
                .foregroundColor(.green)
            }
        }
    }

    // MARK: - Controls

    private var controlsSection: some View {
        VStack(spacing: 16) {
            // Delay selector
            if !engine.isRecording && !countdown.isCountingDown {
                HStack(spacing: 12) {
                    ForEach(delays, id: \.self) { delay in
                        Button {
                            selectedDelay = delay
                        } label: {
                            Text(delay == 0 ? "NOW" : "\(delay)s")
                                .font(.caption.bold())
                                .frame(width: 48, height: 32)
                                .background(selectedDelay == delay ? Color.red.opacity(0.3) : Color.white.opacity(0.1))
                                .foregroundColor(selectedDelay == delay ? .red : .gray)
                                .cornerRadius(8)
                        }
                    }
                }
            }

            // Start / Stop button
            HStack(spacing: 24) {
                // BLE button
                Button {
                    showBLESheet = true
                } label: {
                    Image(systemName: "antenna.radiowaves.left.and.right")
                        .font(.title2)
                        .frame(width: 56, height: 56)
                        .background(ble.connectedDevice != nil ? Color.green.opacity(0.2) : Color.white.opacity(0.1))
                        .foregroundColor(ble.connectedDevice != nil ? .green : .gray)
                        .clipShape(Circle())
                }

                // Main record button
                Button {
                    if engine.isRecording {
                        stopCapture()
                    } else if countdown.isCountingDown {
                        countdown.cancel()
                    } else {
                        startCapture()
                    }
                } label: {
                    ZStack {
                        Circle()
                            .stroke(Color.red, lineWidth: 4)
                            .frame(width: 80, height: 80)
                        if engine.isRecording {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.red)
                                .frame(width: 30, height: 30)
                        } else {
                            Circle()
                                .fill(Color.red)
                                .frame(width: 64, height: 64)
                        }
                    }
                }

                // Share button
                Button {
                    showShareSheet = true
                } label: {
                    Image(systemName: "square.and.arrow.up")
                        .font(.title2)
                        .frame(width: 56, height: 56)
                        .background(Color.white.opacity(0.1))
                        .foregroundColor(.gray)
                        .clipShape(Circle())
                }
                .disabled(engine.isRecording || engine.lastSessionURL == nil)
            }
        }
    }

    // MARK: - Share Sheet

    private var shareSheet: some View {
        NavigationStack {
            VStack(spacing: 20) {
                if let url = engine.lastSessionURL {
                    Text("Session saved to:")
                        .foregroundColor(.secondary)
                    Text(url.lastPathComponent)
                        .font(.headline)
                    Text("Use Files app or iTunes File Sharing to export the session folder.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                } else {
                    Text("No session recorded yet.")
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Export")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") { showShareSheet = false }
                }
            }
        }
    }

    // MARK: - Actions

    private func startCapture() {
        if selectedDelay > 0 {
            countdown.start(delay: selectedDelay) {
                engine.startRecording()
                ScreenKeepAwake.enable()
            }
        } else {
            engine.startRecording()
            ScreenKeepAwake.enable()
        }
    }

    private func stopCapture() {
        engine.stopRecording()
        ScreenKeepAwake.disable()
    }

    // MARK: - Helpers

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%02d:%02d", mins, secs)
    }
}

// MARK: - Sensor Tile

struct SensorTile: View {
    let label: String
    let active: Bool
    let icon: String

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.title3)
            Text(label)
                .font(.caption2.bold())
        }
        .frame(maxWidth: .infinity)
        .frame(height: 56)
        .background(active ? Color.green.opacity(0.15) : Color.white.opacity(0.05))
        .foregroundColor(active ? .green : .gray)
        .cornerRadius(8)
    }
}

// MARK: - Stat Row

struct StatRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.gray)
            Spacer()
            Text(value)
                .font(.body.monospaced())
                .foregroundColor(.white)
        }
    }
}
