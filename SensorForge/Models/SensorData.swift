import Foundation
import simd
import CoreLocation

// MARK: - Unified Timestamp

/// All sensor data is timestamped relative to a shared monotonic clock (ProcessInfo.systemUptime)
/// plus an anchor to wall-clock time captured at session start.
struct SensorTimestamp {
    /// Monotonic time since boot (seconds), from ProcessInfo.processInfo.systemUptime
    let bootTime: TimeInterval
    /// Wall-clock date computed from bootTime + anchor offset
    var wallClock: Date {
        TimestampProvider.shared.wallClock(from: bootTime)
    }
}

// MARK: - ARKit Data

struct ARFrameData {
    let timestamp: SensorTimestamp
    let cameraPose: simd_float4x4
    let cameraIntrinsics: simd_float3x3
    let imageResolution: CGSize
    let trackingState: String
    let ambientIntensity: Float?
    let ambientColorTemperature: Float?
    /// Depth map dimensions (if LiDAR available)
    let depthWidth: Int?
    let depthHeight: Int?
    /// File path for the saved video frame (if writing frames)
    var videoFramePath: String?
    /// File path for the saved depth frame
    var depthFramePath: String?
}

struct MeshAnchorData {
    let timestamp: SensorTimestamp
    let identifier: UUID
    let transform: simd_float4x4
    let vertexCount: Int
    let faceCount: Int
    let classification: String?
}

struct PlaneAnchorData {
    let timestamp: SensorTimestamp
    let identifier: UUID
    let transform: simd_float4x4
    let classification: String
    let extentX: Float
    let extentZ: Float
}

// MARK: - Motion Data

struct IMUData {
    let timestamp: SensorTimestamp
    /// Acceleration in g's (x, y, z)
    let accelerationX: Double
    let accelerationY: Double
    let accelerationZ: Double
    /// Rotation rate in rad/s (x, y, z)
    let rotationRateX: Double
    let rotationRateY: Double
    let rotationRateZ: Double
    /// Attitude (roll, pitch, yaw) in radians
    let roll: Double
    let pitch: Double
    let yaw: Double
    /// Gravity vector
    let gravityX: Double
    let gravityY: Double
    let gravityZ: Double
}

struct MagnetometerData {
    let timestamp: SensorTimestamp
    /// Magnetic field in microteslas
    let x: Double
    let y: Double
    let z: Double
    /// Calibration accuracy (0=uncalibrated, 1=low, 2=medium, 3=high)
    let accuracy: Int32
}

// MARK: - Location Data

struct GPSData {
    let timestamp: SensorTimestamp
    let latitude: Double
    let longitude: Double
    let altitude: Double
    let horizontalAccuracy: Double
    let verticalAccuracy: Double
    let speed: Double
    let course: Double
    let floor: Int?
}

// MARK: - Barometer Data

struct BarometerData {
    let timestamp: SensorTimestamp
    /// Pressure in kilopascals
    let pressure: Double
    /// Relative altitude change in meters (from CMAltimeter)
    let relativeAltitude: Double
}

// MARK: - Audio Data

struct AudioChunkMetadata {
    let timestamp: SensorTimestamp
    let sampleRate: Double
    let channelCount: Int
    let durationSeconds: Double
    let filePath: String
}

// MARK: - BLE Data

struct BLETelemetry {
    let timestamp: SensorTimestamp
    let deviceName: String
    let deviceUUID: String
    let characteristicUUID: String
    let rawHex: String
    let parsedValues: [String: String]
}

// MARK: - Session-Level Containers

/// Thread-safe container for all sensor data collected during a recording session.
@MainActor
final class SensorDataStore {
    var frames: [ARFrameData] = []
    var meshAnchors: [MeshAnchorData] = []
    var planeAnchors: [PlaneAnchorData] = []
    var imuSamples: [IMUData] = []
    var magnetometerSamples: [MagnetometerData] = []
    var gpsSamples: [GPSData] = []
    var barometerSamples: [BarometerData] = []
    var audioChunks: [AudioChunkMetadata] = []
    var bleTelemetry: [BLETelemetry] = []

    var totalSampleCount: Int {
        frames.count + imuSamples.count + magnetometerSamples.count +
        gpsSamples.count + barometerSamples.count + bleTelemetry.count
    }

    func reset() {
        frames.removeAll()
        meshAnchors.removeAll()
        planeAnchors.removeAll()
        imuSamples.removeAll()
        magnetometerSamples.removeAll()
        gpsSamples.removeAll()
        barometerSamples.removeAll()
        audioChunks.removeAll()
        bleTelemetry.removeAll()
    }
}
