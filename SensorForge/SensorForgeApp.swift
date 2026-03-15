import SwiftUI

@main
struct SensorForgeApp: App {
    @StateObject private var engine = CaptureEngine()
    @StateObject private var ble = BLEBridge()
    @StateObject private var ugv = WiFiBridge()
    @StateObject private var shakeDetector = ShakeDetector()

    var body: some Scene {
        WindowGroup {
            MainCaptureView(engine: engine, ble: ble, ugv: ugv)
                .onAppear {
                    engine.setBLEBridge(ble)
                    engine.setWiFiBridge(ugv)

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
