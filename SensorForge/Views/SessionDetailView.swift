import SwiftUI

struct SessionDetailView: View {
    let session: SessionMetadata

    @State private var showShareSheet = false

    var body: some View {
        List {
            Section("Overview") {
                row("Date", value: session.startDate.formatted(date: .long, time: .shortened))
                row("Duration", value: formatDuration(session.durationSeconds ?? 0))
                row("Device", value: session.deviceModel)
                row("OS", value: session.osVersion)
                row("Start Method", value: session.startMethod.rawValue.capitalized)
                if let size = session.totalFileSizeBytes {
                    row("Size", value: formatBytes(size))
                }
            }

            Section("Sensor Data") {
                row("Camera Frames", value: "\(session.frameCount)")
                row("IMU Samples", value: "\(session.imuSampleCount)")
                row("Magnetometer", value: "\(session.magnetometerSampleCount)")
                row("GPS Points", value: "\(session.gpsSampleCount)")
                row("Barometer", value: "\(session.barometerSampleCount)")
                row("BLE Telemetry", value: "\(session.bleTelemetryCount)")
            }

            Section("Capabilities") {
                capabilityRow("ARKit", available: session.hasARKit)
                capabilityRow("LiDAR", available: session.hasLiDAR)
                capabilityRow("GPS", available: session.hasGPS)
                capabilityRow("BLE Devices", available: session.hasBLE)
            }

            Section("Export") {
                Button {
                    showShareSheet = true
                } label: {
                    Label("Share Session Files", systemImage: "square.and.arrow.up")
                }

                NavigationLink {
                    fileListView
                } label: {
                    Label("Browse Files", systemImage: "folder")
                }
            }
        }
        .navigationTitle("Session Details")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showShareSheet) {
            let dir = SessionStore.shared.sessionDirectory(for: session)
            ShareSheet(activityItems: [dir])
        }
    }

    private func row(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
    }

    private func capabilityRow(_ label: String, available: Bool) -> some View {
        HStack {
            Text(label)
            Spacer()
            Image(systemName: available ? "checkmark.circle.fill" : "xmark.circle")
                .foregroundColor(available ? .green : .gray)
        }
    }

    private var fileListView: some View {
        let dir = SessionStore.shared.sessionDirectory(for: session)
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey]
        )) ?? []

        return List(files, id: \.absoluteString) { url in
            HStack {
                Image(systemName: iconForFile(url))
                    .foregroundColor(.accentColor)
                VStack(alignment: .leading) {
                    Text(url.lastPathComponent)
                        .font(.body)
                    if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                        Text(formatBytes(Int64(size)))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .navigationTitle("Files")
    }

    private func iconForFile(_ url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "csv": return "tablecells"
        case "json": return "doc.text"
        case "mp4", "mov": return "video"
        case "wav", "m4a": return "waveform"
        default: return "doc"
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return String(format: "%d:%02d", m, s)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
