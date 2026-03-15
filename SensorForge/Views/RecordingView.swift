import SwiftUI

struct RecordingView: View {
    @EnvironmentObject var coordinator: RecordingCoordinator
    @EnvironmentObject var countdownTimer: CountdownTimer
    @EnvironmentObject var shakeDetector: ShakeDetector

    @State private var showCountdownPicker = false
    @State private var selectedCountdown = 10

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                VStack(spacing: 32) {
                    // Status header
                    statusHeader

                    Spacer()

                    // Sensor status grid
                    SensorStatusView()
                        .environmentObject(coordinator)

                    Spacer()

                    // Countdown display
                    if countdownTimer.isCountingDown {
                        countdownDisplay
                    }

                    // Record button
                    recordButton

                    // Start options
                    if !coordinator.isRecording && !countdownTimer.isCountingDown {
                        startOptions
                    }

                    // Recording stats
                    if coordinator.isRecording {
                        recordingStats
                    }
                }
                .padding()
            }
            .navigationTitle("SensorForge")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
        }
    }

    // MARK: - Status Header

    private var statusHeader: some View {
        HStack {
            Circle()
                .fill(coordinator.isRecording ? Color.red : Color.gray)
                .frame(width: 12, height: 12)
                .overlay {
                    if coordinator.isRecording {
                        Circle()
                            .fill(Color.red.opacity(0.4))
                            .frame(width: 24, height: 24)
                            .animation(.easeInOut(duration: 1).repeatForever(), value: coordinator.isRecording)
                    }
                }

            Text(coordinator.isRecording ? "RECORDING" : "READY")
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(coordinator.isRecording ? .red : .gray)
                .tracking(2)

            Spacer()

            if coordinator.isRecording {
                Text(formatDuration(coordinator.recordingDuration))
                    .font(.system(.title2, design: .monospaced))
                    .foregroundColor(.white)
            }
        }
    }

    // MARK: - Countdown Display

    private var countdownDisplay: some View {
        VStack(spacing: 8) {
            Text("\(countdownTimer.remainingSeconds)")
                .font(.system(size: 96, weight: .ultraLight, design: .monospaced))
                .foregroundColor(.orange)

            Text("Recording starts in...")
                .font(.caption)
                .foregroundColor(.orange.opacity(0.7))

            Button("Cancel") {
                countdownTimer.cancel()
            }
            .foregroundColor(.orange)
            .padding(.top, 4)
        }
    }

    // MARK: - Record Button

    private var recordButton: some View {
        Button {
            if coordinator.isRecording {
                coordinator.stopRecording()
            } else if countdownTimer.isCountingDown {
                countdownTimer.cancel()
            } else {
                coordinator.startRecording(method: .tap)
            }
        } label: {
            ZStack {
                Circle()
                    .strokeBorder(Color.white, lineWidth: 4)
                    .frame(width: 80, height: 80)

                if coordinator.isRecording {
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
        .disabled(countdownTimer.isCountingDown)
    }

    // MARK: - Start Options

    private var startOptions: some View {
        HStack(spacing: 24) {
            // Countdown start
            Button {
                showCountdownPicker.toggle()
            } label: {
                VStack(spacing: 4) {
                    Image(systemName: "timer")
                        .font(.title3)
                    Text("Timer")
                        .font(.caption2)
                }
                .foregroundColor(.white.opacity(0.7))
            }
            .sheet(isPresented: $showCountdownPicker) {
                countdownPickerSheet
            }

            // Shake indicator
            VStack(spacing: 4) {
                Image(systemName: "iphone.radiowaves.left.and.right")
                    .font(.title3)
                Text("Shake x3")
                    .font(.caption2)
            }
            .foregroundColor(shakeDetector.isEnabled ? .white.opacity(0.7) : .white.opacity(0.3))
        }
    }

    // MARK: - Recording Stats

    private var recordingStats: some View {
        HStack(spacing: 16) {
            statBadge(label: "Samples", value: "\(coordinator.sampleCount)")
            statBadge(label: "FPS", value: String(format: "%.0f", coordinator.arkitManager.currentFPS))
            statBadge(label: "IMU Hz", value: String(format: "%.0f", coordinator.motionManager.currentIMURate))
        }
    }

    private func statBadge(label: String, value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(.body, design: .monospaced))
                .fontWeight(.semibold)
                .foregroundColor(.white)
            Text(label)
                .font(.caption2)
                .foregroundColor(.gray)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.white.opacity(0.1))
        .cornerRadius(8)
    }

    // MARK: - Countdown Picker

    private var countdownPickerSheet: some View {
        NavigationStack {
            List {
                ForEach(CountdownTimer.presets, id: \.self) { seconds in
                    Button {
                        showCountdownPicker = false
                        countdownTimer.start(seconds: seconds)
                    } label: {
                        HStack {
                            Text("\(seconds) seconds")
                            Spacer()
                            if seconds == selectedCountdown {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            }
            .navigationTitle("Countdown Timer")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { showCountdownPicker = false }
                }
            }
        }
        .presentationDetents([.medium])
    }

    // MARK: - Helpers

    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        let tenths = Int((duration - floor(duration)) * 10)
        return String(format: "%02d:%02d.%d", minutes, seconds, tenths)
    }
}
