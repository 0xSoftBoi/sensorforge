import SwiftUI

@main
struct SensorForgeApp: App {
    @StateObject private var engine = CaptureEngine()
    @StateObject private var ble = BLEBridge()
    @StateObject private var bridge = WiFiBridge()
    @StateObject private var shakeDetector = ShakeDetector()

    var body: some Scene {
        WindowGroup {
            MainCaptureView(engine: engine, ble: ble, bridge: bridge)
                .onAppear {
                    engine.setBLEBridge(ble)
                    bridge.startBrowsing()

                    // Enable shake-to-toggle recording
                    shakeDetector.start {
                        if engine.isRecording {
                            engine.stopRecording()
                            ScreenKeepAwake.disable()
                        } else {
                            engine.startRecording()
                            ScreenKeepAwake.enable()
                        }
                    }
                }
                .onDisappear {
                    shakeDetector.stop()
                }
        }
    }
}
