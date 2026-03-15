import SwiftUI

@main
struct SensorForgeApp: App {
    @StateObject private var coordinator = RecordingCoordinator()
    @StateObject private var shakeDetector = ShakeDetector()
    @StateObject private var countdownTimer = CountdownTimer()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(coordinator)
                .environmentObject(shakeDetector)
                .environmentObject(countdownTimer)
                .onAppear {
                    coordinator.requestPermissions()
                    setupShakeDetector()
                    setupCountdown()
                }
        }
    }

    private func setupShakeDetector() {
        shakeDetector.onTripleShake = { [weak coordinator] in
            guard let coordinator else { return }
            Task { @MainActor in
                if coordinator.isRecording {
                    coordinator.stopRecording()
                } else {
                    coordinator.startRecording(method: .shake)
                }
            }
        }
    }

    private func setupCountdown() {
        countdownTimer.onComplete = { [weak coordinator] in
            Task { @MainActor in
                coordinator?.startRecording(method: .countdown)
            }
        }
    }
}
