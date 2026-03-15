import Foundation
import CoreMotion
import Combine

/// Captures IMU (accelerometer + gyroscope) and magnetometer data via CoreMotion.
final class MotionManager: ObservableObject {
    @Published var isRunning = false
    @Published var currentIMURate: Double = 0
    @Published var latestAcceleration: (x: Double, y: Double, z: Double) = (0, 0, 0)

    private let motionManager = CMMotionManager()
    private var dataStore: SensorDataStore?
    private var imuTimer: Timer?
    private var magTimer: Timer?

    /// Target IMU update interval (200 Hz)
    private let imuInterval: TimeInterval = 1.0 / 200.0
    /// Target magnetometer update interval (50 Hz)
    private let magInterval: TimeInterval = 1.0 / 50.0

    private var sampleCount = 0
    private var sampleTimer: TimeInterval = 0

    var isAccelerometerAvailable: Bool { motionManager.isAccelerometerAvailable }
    var isGyroAvailable: Bool { motionManager.isGyroAvailable }
    var isMagnetometerAvailable: Bool { motionManager.isMagnetometerAvailable }

    func start(dataStore: SensorDataStore) {
        self.dataStore = dataStore
        sampleCount = 0
        sampleTimer = ProcessInfo.processInfo.systemUptime

        startDeviceMotion()
        startMagnetometer()
        isRunning = true
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
        motionManager.stopMagnetometerUpdates()
        imuTimer?.invalidate()
        magTimer?.invalidate()
        isRunning = false
    }

    // MARK: - Device Motion (fused accel + gyro + attitude)

    private func startDeviceMotion() {
        guard motionManager.isDeviceMotionAvailable else { return }

        motionManager.deviceMotionUpdateInterval = imuInterval
        motionManager.startDeviceMotionUpdates(using: .xArbitraryCorrectedZVertical, to: .main) { [weak self] motion, error in
            guard let self, let motion else { return }

            let ts = SensorTimestamp(bootTime: motion.timestamp)

            let data = IMUData(
                timestamp: ts,
                accelerationX: motion.userAcceleration.x,
                accelerationY: motion.userAcceleration.y,
                accelerationZ: motion.userAcceleration.z,
                rotationRateX: motion.rotationRate.x,
                rotationRateY: motion.rotationRate.y,
                rotationRateZ: motion.rotationRate.z,
                roll: motion.attitude.roll,
                pitch: motion.attitude.pitch,
                yaw: motion.attitude.yaw,
                gravityX: motion.gravity.x,
                gravityY: motion.gravity.y,
                gravityZ: motion.gravity.z
            )

            self.latestAcceleration = (motion.userAcceleration.x, motion.userAcceleration.y, motion.userAcceleration.z)

            // FPS counter
            self.sampleCount += 1
            let now = ProcessInfo.processInfo.systemUptime
            if now - self.sampleTimer >= 1.0 {
                self.currentIMURate = Double(self.sampleCount)
                self.sampleCount = 0
                self.sampleTimer = now
            }

            Task { @MainActor in
                self.dataStore?.imuSamples.append(data)
            }
        }
    }

    // MARK: - Magnetometer

    private func startMagnetometer() {
        guard motionManager.isMagnetometerAvailable else { return }

        motionManager.magnetometerUpdateInterval = magInterval
        motionManager.startMagnetometerUpdates(to: .main) { [weak self] magnetData, error in
            guard let self, let magnetData else { return }

            let ts = SensorTimestamp(bootTime: magnetData.timestamp)

            let data = MagnetometerData(
                timestamp: ts,
                x: magnetData.magneticField.x,
                y: magnetData.magneticField.y,
                z: magnetData.magneticField.z,
                accuracy: -1  // Raw magnetometer doesn't provide accuracy; use heading accuracy if needed
            )

            Task { @MainActor in
                self.dataStore?.magnetometerSamples.append(data)
            }
        }
    }
}
