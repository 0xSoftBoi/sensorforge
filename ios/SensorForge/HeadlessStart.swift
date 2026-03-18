import Foundation
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
                AudioServicesPlaySystemSound(1057)
            } else {
                // Vibrate + chime on start
                AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
                AudioServicesPlaySystemSound(1025)
                self.timer?.invalidate()
                self.timer = nil
                self.isCountingDown = false
                let completion = self.onComplete
                self.onComplete = nil
                completion?()
            }
        }
    }

    func cancel() {
        timer?.invalidate()
        timer = nil
        isCountingDown = false
        secondsRemaining = 0
        onComplete = nil
    }
}

// MARK: - Shake Detector

/// Detects shake gestures via UIDevice motion events to toggle recording.
/// Does NOT use its own CMMotionManager — avoids conflict with CaptureEngine.
/// Uses the standard UIResponder shake gesture (motionBegan/motionEnded).
final class ShakeDetector: ObservableObject {
    @Published var isEnabled = true

    private var onToggle: (() -> Void)?
    private var lastShakeTime: Date = .distantPast
    private let cooldown: TimeInterval = 2.0  // prevent double-triggers

    func start(onToggle: @escaping () -> Void) {
        self.onToggle = onToggle
        // Enable shake detection via UIDevice
        UIDevice.current.isProximityMonitoringEnabled = false
        // The actual shake detection happens in ShakeDetectingWindow
        NotificationCenter.default.addObserver(
            self, selector: #selector(handleShake),
            name: .deviceDidShake, object: nil
        )
    }

    func stop() {
        NotificationCenter.default.removeObserver(self, name: .deviceDidShake, object: nil)
        onToggle = nil
    }

    @objc private func handleShake() {
        guard isEnabled else { return }
        let now = Date()
        guard now.timeIntervalSince(lastShakeTime) > cooldown else { return }
        lastShakeTime = now
        AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
        onToggle?()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}

/// Custom UIWindow subclass that intercepts shake gestures and posts a notification.
/// Set this as the app's window class to enable shake detection without CMMotionManager.
class ShakeDetectingWindow: UIWindow {
    override func motionBegan(_ motion: UIEvent.EventSubtype, with event: UIEvent?) {
        if motion == .motionShake {
            NotificationCenter.default.post(name: .deviceDidShake, object: nil)
        }
        super.motionBegan(motion, with: event)
    }
}

extension Notification.Name {
    static let deviceDidShake = Notification.Name("com.sensorforge.deviceDidShake")
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

// MARK: - Notification Names (Siri)

extension Notification.Name {
    static let siriStartRecording = Notification.Name("com.sensorforge.siri.start")
    static let siriStopRecording = Notification.Name("com.sensorforge.siri.stop")
}
