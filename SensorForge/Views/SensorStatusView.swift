import SwiftUI

/// Displays the live status of all sensors in a grid.
struct SensorStatusView: View {
    @EnvironmentObject var coordinator: RecordingCoordinator

    var body: some View {
        LazyVGrid(columns: [
            GridItem(.flexible()),
            GridItem(.flexible()),
            GridItem(.flexible())
        ], spacing: 12) {
            sensorTile(
                icon: "camera.fill",
                label: "Camera",
                status: coordinator.arkitManager.isRunning ? "Active" : "Ready",
                active: coordinator.arkitManager.isRunning,
                detail: coordinator.arkitManager.isRunning
                    ? "\(String(format: "%.0f", coordinator.arkitManager.currentFPS)) fps"
                    : nil
            )

            sensorTile(
                icon: "lidar.scanner",
                label: "LiDAR",
                status: ARKitManager.isLiDARAvailable
                    ? (coordinator.arkitManager.isRunning ? "Active" : "Ready")
                    : "N/A",
                active: coordinator.arkitManager.isRunning && ARKitManager.isLiDARAvailable
            )

            sensorTile(
                icon: "gyroscope",
                label: "IMU",
                status: coordinator.motionManager.isRunning ? "Active" : "Ready",
                active: coordinator.motionManager.isRunning,
                detail: coordinator.motionManager.isRunning
                    ? "\(String(format: "%.0f", coordinator.motionManager.currentIMURate)) Hz"
                    : nil
            )

            sensorTile(
                icon: "location.fill",
                label: "GPS",
                status: coordinator.locationManager.isRunning ? "Active" : "Ready",
                active: coordinator.locationManager.isRunning
            )

            sensorTile(
                icon: "barometer",
                label: "Baro",
                status: coordinator.barometerManager.isAvailable
                    ? (coordinator.barometerManager.isRunning ? "Active" : "Ready")
                    : "N/A",
                active: coordinator.barometerManager.isRunning
            )

            sensorTile(
                icon: "mic.fill",
                label: "Audio",
                status: coordinator.audioCaptureManager.isRunning ? "Active" : "Ready",
                active: coordinator.audioCaptureManager.isRunning
            )

            sensorTile(
                icon: "antenna.radiowaves.left.and.right",
                label: "BLE",
                status: !coordinator.bleBridge.connectedDevices.isEmpty
                    ? "\(coordinator.bleBridge.connectedDevices.count) dev"
                    : "No devices",
                active: !coordinator.bleBridge.connectedDevices.isEmpty
            )

            sensorTile(
                icon: "compass.drawing",
                label: "Mag",
                status: coordinator.motionManager.isMagnetometerAvailable
                    ? (coordinator.motionManager.isRunning ? "Active" : "Ready")
                    : "N/A",
                active: coordinator.motionManager.isRunning && coordinator.motionManager.isMagnetometerAvailable
            )

            sensorTile(
                icon: "light.max",
                label: "Light",
                status: coordinator.arkitManager.isRunning ? "Active" : "Ready",
                active: coordinator.arkitManager.isRunning
            )
        }
    }

    private func sensorTile(icon: String, label: String, status: String, active: Bool, detail: String? = nil) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(active ? .green : .gray)

            Text(label)
                .font(.caption2)
                .fontWeight(.medium)
                .foregroundColor(.white)

            Text(detail ?? status)
                .font(.system(.caption2, design: .monospaced))
                .foregroundColor(active ? .green.opacity(0.8) : .gray.opacity(0.6))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(active ? Color.green.opacity(0.1) : Color.white.opacity(0.05))
        .cornerRadius(10)
    }
}
