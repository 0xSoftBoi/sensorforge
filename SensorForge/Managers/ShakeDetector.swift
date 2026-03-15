import Foundation
import UIKit

/// Detects 3 shakes in rapid succession to toggle recording.
///
/// Uses UIDevice motion events. The user shakes the phone 3 times within
/// a 2-second window to trigger the callback.
final class ShakeDetector: ObservableObject {
    @Published var isEnabled = true
    @Published var shakeCount = 0

    var onTripleShake: (() -> Void)?

    private var shakeTimes: [TimeInterval] = []
    private let requiredShakes = 3
    private let timeWindow: TimeInterval = 2.0

    func recordShake() {
        guard isEnabled else { return }

        let now = ProcessInfo.processInfo.systemUptime
        shakeTimes.append(now)

        // Remove old shakes outside the window
        shakeTimes = shakeTimes.filter { now - $0 <= timeWindow }

        shakeCount = shakeTimes.count

        if shakeTimes.count >= requiredShakes {
            shakeTimes.removeAll()
            shakeCount = 0
            onTripleShake?()
        }
    }
}

/// A UIWindow subclass that detects shake gestures and forwards them to ShakeDetector.
final class ShakeDetectingWindow: UIWindow {
    var shakeDetector: ShakeDetector?

    override func motionEnded(_ motion: UIEvent.EventSubtype, with event: UIEvent?) {
        if motion == .motionShake {
            shakeDetector?.recordShake()
        }
        super.motionEnded(motion, with: event)
    }
}
