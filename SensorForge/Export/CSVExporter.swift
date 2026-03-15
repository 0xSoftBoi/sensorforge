import Foundation
import simd

/// Exports all sensor data to CSV files within a session directory.
enum CSVExporter {

    static func export(dataStore: SensorDataStore, metadata: SessionMetadata, to directory: URL) {
        exportFrames(dataStore.frames, to: directory)
        exportIMU(dataStore.imuSamples, to: directory)
        exportMagnetometer(dataStore.magnetometerSamples, to: directory)
        exportGPS(dataStore.gpsSamples, to: directory)
        exportBarometer(dataStore.barometerSamples, to: directory)
        exportPlanes(dataStore.planeAnchors, to: directory)
        exportMeshes(dataStore.meshAnchors, to: directory)
        exportBLE(dataStore.bleTelemetry, to: directory)
        exportMetadata(metadata, to: directory)
    }

    // MARK: - CSV Escaping

    /// Escapes a string value for safe CSV inclusion, preventing CSV injection attacks.
    /// Handles formula injection (=, +, -, @, \t, \r) and proper quoting of special characters.
    private static func escapeCSV(_ value: String) -> String {
        let formulaPrefixes: [Character] = ["=", "+", "-", "@", "\t", "\r"]
        let needsQuoting = value.contains(",") || value.contains("\"") || value.contains("\n") ||
                           value.contains("\r") || formulaPrefixes.contains(where: { value.hasPrefix(String($0)) })

        if !needsQuoting { return value }

        var escaped = value.replacingOccurrences(of: "\"", with: "\"\"")

        // Neutralize formula injection by prepending a single quote inside quotes
        if let first = escaped.first, formulaPrefixes.contains(first) {
            escaped = "'" + escaped
        }

        return "\"\(escaped)\""
    }

    // MARK: - Camera Frames

    private static func exportFrames(_ frames: [ARFrameData], to dir: URL) {
        guard !frames.isEmpty else { return }

        var csv = "session_time_s,boot_time_s,wall_clock,tracking_state,"
        csv += "pose_m00,pose_m01,pose_m02,pose_m03,"
        csv += "pose_m10,pose_m11,pose_m12,pose_m13,"
        csv += "pose_m20,pose_m21,pose_m22,pose_m23,"
        csv += "pose_m30,pose_m31,pose_m32,pose_m33,"
        csv += "position_x,position_y,position_z,"
        csv += "image_width,image_height,"
        csv += "ambient_intensity,ambient_color_temp,"
        csv += "depth_width,depth_height\n"

        let tp = TimestampProvider.shared
        let df = ISO8601DateFormatter()

        for frame in frames {
            let st = tp.sessionTime(from: frame.timestamp.bootTime)
            let wc = df.string(from: frame.timestamp.wallClock)
            let m = frame.cameraPose

            csv += "\(st),\(frame.timestamp.bootTime),\(wc),\(escapeCSV(frame.trackingState)),"
            csv += "\(m[0][0]),\(m[0][1]),\(m[0][2]),\(m[0][3]),"
            csv += "\(m[1][0]),\(m[1][1]),\(m[1][2]),\(m[1][3]),"
            csv += "\(m[2][0]),\(m[2][1]),\(m[2][2]),\(m[2][3]),"
            csv += "\(m[3][0]),\(m[3][1]),\(m[3][2]),\(m[3][3]),"
            csv += "\(m[3][0]),\(m[3][1]),\(m[3][2]),"
            csv += "\(Int(frame.imageResolution.width)),\(Int(frame.imageResolution.height)),"
            csv += "\(frame.ambientIntensity ?? 0),\(frame.ambientColorTemperature ?? 0),"
            csv += "\(frame.depthWidth ?? 0),\(frame.depthHeight ?? 0)\n"
        }

        write(csv, to: dir.appendingPathComponent("camera_poses.csv"))
    }

    // MARK: - IMU

    private static func exportIMU(_ samples: [IMUData], to dir: URL) {
        guard !samples.isEmpty else { return }

        var csv = "session_time_s,boot_time_s,"
        csv += "accel_x_g,accel_y_g,accel_z_g,"
        csv += "gyro_x_rads,gyro_y_rads,gyro_z_rads,"
        csv += "roll_rad,pitch_rad,yaw_rad,"
        csv += "gravity_x,gravity_y,gravity_z\n"

        let tp = TimestampProvider.shared

        for s in samples {
            let st = tp.sessionTime(from: s.timestamp.bootTime)
            csv += "\(st),\(s.timestamp.bootTime),"
            csv += "\(s.accelerationX),\(s.accelerationY),\(s.accelerationZ),"
            csv += "\(s.rotationRateX),\(s.rotationRateY),\(s.rotationRateZ),"
            csv += "\(s.roll),\(s.pitch),\(s.yaw),"
            csv += "\(s.gravityX),\(s.gravityY),\(s.gravityZ)\n"
        }

        write(csv, to: dir.appendingPathComponent("imu.csv"))
    }

    // MARK: - Magnetometer

    private static func exportMagnetometer(_ samples: [MagnetometerData], to dir: URL) {
        guard !samples.isEmpty else { return }

        var csv = "session_time_s,boot_time_s,mag_x_ut,mag_y_ut,mag_z_ut,accuracy\n"
        let tp = TimestampProvider.shared

        for s in samples {
            let st = tp.sessionTime(from: s.timestamp.bootTime)
            csv += "\(st),\(s.timestamp.bootTime),\(s.x),\(s.y),\(s.z),\(s.accuracy)\n"
        }

        write(csv, to: dir.appendingPathComponent("magnetometer.csv"))
    }

    // MARK: - GPS

    private static func exportGPS(_ samples: [GPSData], to dir: URL) {
        guard !samples.isEmpty else { return }

        var csv = "session_time_s,boot_time_s,latitude,longitude,altitude_m,"
        csv += "horizontal_accuracy_m,vertical_accuracy_m,speed_ms,course_deg,floor\n"
        let tp = TimestampProvider.shared

        for s in samples {
            let st = tp.sessionTime(from: s.timestamp.bootTime)
            csv += "\(st),\(s.timestamp.bootTime),"
            csv += "\(s.latitude),\(s.longitude),\(s.altitude),"
            csv += "\(s.horizontalAccuracy),\(s.verticalAccuracy),"
            csv += "\(s.speed),\(s.course),\(s.floor ?? -1)\n"
        }

        write(csv, to: dir.appendingPathComponent("gps.csv"))
    }

    // MARK: - Barometer

    private static func exportBarometer(_ samples: [BarometerData], to dir: URL) {
        guard !samples.isEmpty else { return }

        var csv = "session_time_s,boot_time_s,pressure_kpa,relative_altitude_m\n"
        let tp = TimestampProvider.shared

        for s in samples {
            let st = tp.sessionTime(from: s.timestamp.bootTime)
            csv += "\(st),\(s.timestamp.bootTime),\(s.pressure),\(s.relativeAltitude)\n"
        }

        write(csv, to: dir.appendingPathComponent("barometer.csv"))
    }

    // MARK: - Planes

    private static func exportPlanes(_ planes: [PlaneAnchorData], to dir: URL) {
        guard !planes.isEmpty else { return }

        var csv = "session_time_s,boot_time_s,identifier,classification,extent_x,extent_z,"
        csv += "position_x,position_y,position_z\n"
        let tp = TimestampProvider.shared

        for p in planes {
            let st = tp.sessionTime(from: p.timestamp.bootTime)
            csv += "\(st),\(p.timestamp.bootTime),\(p.identifier),\(escapeCSV(p.classification)),"
            csv += "\(p.extentX),\(p.extentZ),"
            csv += "\(p.transform[3][0]),\(p.transform[3][1]),\(p.transform[3][2])\n"
        }

        write(csv, to: dir.appendingPathComponent("planes.csv"))
    }

    // MARK: - Meshes

    private static func exportMeshes(_ meshes: [MeshAnchorData], to dir: URL) {
        guard !meshes.isEmpty else { return }

        var csv = "session_time_s,boot_time_s,identifier,classification,vertex_count,face_count,"
        csv += "position_x,position_y,position_z\n"
        let tp = TimestampProvider.shared

        for m in meshes {
            let st = tp.sessionTime(from: m.timestamp.bootTime)
            let classStr = escapeCSV(m.classification ?? "none")
            csv += "\(st),\(m.timestamp.bootTime),\(m.identifier),\(classStr),\(m.vertexCount),\(m.faceCount),"
            csv += "\(m.transform[3][0]),\(m.transform[3][1]),\(m.transform[3][2])\n"
        }

        write(csv, to: dir.appendingPathComponent("meshes.csv"))
    }

    // MARK: - BLE Telemetry

    private static func exportBLE(_ telemetry: [BLETelemetry], to dir: URL) {
        guard !telemetry.isEmpty else { return }

        var csv = "session_time_s,boot_time_s,device_name,device_uuid,characteristic_uuid,raw_hex,parsed_values\n"
        let tp = TimestampProvider.shared

        for t in telemetry {
            let st = tp.sessionTime(from: t.timestamp.bootTime)
            let parsedStr = t.parsedValues.map { "\($0.key)=\($0.value)" }.joined(separator: ";")
            csv += "\(st),\(t.timestamp.bootTime),\(escapeCSV(t.deviceName)),\(escapeCSV(t.deviceUUID)),"
            csv += "\(escapeCSV(t.characteristicUUID)),\(escapeCSV(t.rawHex)),\(escapeCSV(parsedStr))\n"
        }

        write(csv, to: dir.appendingPathComponent("ble_telemetry.csv"))
    }

    // MARK: - Session Metadata

    private static func exportMetadata(_ metadata: SessionMetadata, to dir: URL) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            let data = try encoder.encode(metadata)
            try data.write(to: dir.appendingPathComponent("session_metadata.json"), options: .atomic)
        } catch {
            print("[CSVExporter] Failed to export metadata: \(error)")
        }
    }

    // MARK: - Helpers

    private static func write(_ content: String, to url: URL) {
        do {
            try content.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            print("[CSVExporter] Failed to write \(url.lastPathComponent): \(error)")
        }
    }
}
