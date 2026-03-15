import Foundation
import Combine

/// Manages a countdown before recording starts.
/// Gives the user time to mount the phone on a robot after tapping start.
@MainActor
final class CountdownTimer: ObservableObject {
    @Published var isCountingDown = false
    @Published var remainingSeconds: Int = 0

    private var timer: Timer?

    var onComplete: (() -> Void)?

    /// Available countdown presets in seconds.
    static let presets: [Int] = [5, 10, 15, 30, 60]

    func start(seconds: Int) {
        guard !isCountingDown else { return }
        remainingSeconds = seconds
        isCountingDown = true

        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.remainingSeconds -= 1
                if self.remainingSeconds <= 0 {
                    self.cancel()
                    self.onComplete?()
                }
            }
        }
    }

    func cancel() {
        timer?.invalidate()
        timer = nil
        isCountingDown = false
        remainingSeconds = 0
    }
}
