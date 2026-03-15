import Foundation
import CoreMotion
import UIKit
import AVFoundation
import AppIntents

// MARK: - Countdown Starter

/// Plays countdown with audio ticks and vibration, then triggers recording.
final class CountdownStarter: ObservableObject {
    @Published var secondsRemaining: Int = 0
    @Published var isCountingDown = false

    private var timer: Timer?
    private var onComplete: (() -> Void)?

    func start(delay: Int, onComplete: @escaping () -> Void) {
        guard !isCountingDown else { return }
        self.onComplete = onComplete
        secondsRemaining = delay
        isCountingDown = true

        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.secondsRemaining -= 1

            if self.secondsRemaining > 0 {
                // Tick sound
                AudioServicesPlaySystemSound(1057)
            } else {
                // Triple vibrate on start
                for i in 0..<3 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * 0.2) {
                        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
                    }
                }
                AudioServicesPlaySystemSound(1025)  // Start chime
                self.timer?.invalidate()
                self.timer = nil
                self.isCountingDown = false
                self.onComplete?()
            }
        }
    }

    func cancel() {
        timer?.invalidate()
        timer = nil
        isCountingDown = false
        secondsRemaining = 0
    }
}

// MARK: - Shake Detector

/// Detects 3 shakes within 2 seconds at >2.5G to toggle recording.
/// Works when phone is mounted and screen may be hard to reach.
final class ShakeDetector: ObservableObject {
    @Published var isEnabled = true

    private let motionManager = CMMotionManager()
    private var shakeTimes: [Date] = []
    private let shakeThreshold: Double = 2.5  // G-force
    private let shakeWindow: TimeInterval = 2.0
    private let requiredShakes = 3
    private var onToggle: (() -> Void)?

    func start(onToggle: @escaping () -> Void) {
        self.onToggle = onToggle
        guard motionManager.isAccelerometerAvailable else { return }

        motionManager.accelerometerUpdateInterval = 1.0 / 50.0
        motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, _ in
            guard let self, self.isEnabled, let data else { return }
            let magnitude = sqrt(
                data.acceleration.x * data.acceleration.x +
                data.acceleration.y * data.acceleration.y +
                data.acceleration.z * data.acceleration.z
            )

            if magnitude > self.shakeThreshold {
                let now = Date()
                self.shakeTimes.append(now)
                // Remove old shakes outside window
                self.shakeTimes = self.shakeTimes.filter {
                    now.timeIntervalSince($0) < self.shakeWindow
                }

                if self.shakeTimes.count >= self.requiredShakes {
                    self.shakeTimes.removeAll()
                    AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
                    self.onToggle?()
                }
            }
        }
    }

    func stop() {
        motionManager.stopAccelerometerUpdates()
    }
}

// MARK: - Screen Keep Awake

/// Prevents screen from dimming during recording.
final class ScreenKeepAwake {
    static func enable() {
        DispatchQueue.main.async {
            UIApplication.shared.isIdleTimerDisabled = true
        }
    }

    static func disable() {
        DispatchQueue.main.async {
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }
}

// MARK: - Siri Shortcuts (AppIntents)

@available(iOS 16.0, *)
struct StartRecordingIntent: AppIntent {
    static var title: LocalizedStringResource = "Start SensorForge Recording"
    static var description: IntentDescription = "Starts recording all sensors"
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult {
        // Post notification that the app will pick up
        NotificationCenter.default.post(name: .siriStartRecording, object: nil)
        return .result()
    }
}

@available(iOS 16.0, *)
struct StopRecordingIntent: AppIntent {
    static var title: LocalizedStringResource = "Stop SensorForge Recording"
    static var description: IntentDescription = "Stops the current recording"
    static var openAppWhenRun: Bool = true

    func perform() async throws -> some IntentResult {
        NotificationCenter.default.post(name: .siriStopRecording, object: nil)
        return .result()
    }
}

@available(iOS 16.0, *)
struct SensorForgeShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: StartRecordingIntent(),
            phrases: [
                "Start \(.applicationName)",
                "Start recording with \(.applicationName)",
                "\(.applicationName) start",
            ],
            shortTitle: "Start Recording",
            systemImageName: "record.circle"
        )
        AppShortcut(
            intent: StopRecordingIntent(),
            phrases: [
                "Stop \(.applicationName)",
                "Stop recording with \(.applicationName)",
                "\(.applicationName) stop",
            ],
            shortTitle: "Stop Recording",
            systemImageName: "stop.circle"
        )
    }
}

// MARK: - Notification Names

extension Notification.Name {
    static let siriStartRecording = Notification.Name("com.sensorforge.siri.start")
    static let siriStopRecording = Notification.Name("com.sensorforge.siri.stop")
}
