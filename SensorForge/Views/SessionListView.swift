import SwiftUI

struct SessionListView: View {
    @State private var sessions: [SessionMetadata] = []

    var body: some View {
        NavigationStack {
            Group {
                if sessions.isEmpty {
                    ContentUnavailableView(
                        "No Sessions Yet",
                        systemImage: "waveform.path",
                        description: Text("Record your first session to see it here.")
                    )
                } else {
                    List {
                        ForEach(sessions) { session in
                            NavigationLink(destination: SessionDetailView(session: session)) {
                                sessionRow(session)
                            }
                        }
                        .onDelete(perform: deleteSessions)
                    }
                }
            }
            .navigationTitle("Sessions")
            .onAppear { sessions = SessionStore.shared.loadSessions() }
            .refreshable { sessions = SessionStore.shared.loadSessions() }
            .toolbar {
                if !sessions.isEmpty {
                    EditButton()
                }
            }
        }
    }

    private func sessionRow(_ session: SessionMetadata) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(session.startDate, style: .date)
                    .font(.headline)
                Text(session.startDate, style: .time)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 12) {
                Label("\(formatDuration(session.durationSeconds ?? 0))", systemImage: "clock")
                Label("\(session.frameCount) frames", systemImage: "camera")
                Label("\(session.imuSampleCount) IMU", systemImage: "gyroscope")
            }
            .font(.caption)
            .foregroundColor(.secondary)

            HStack(spacing: 8) {
                if session.hasLiDAR {
                    badge("LiDAR", color: .blue)
                }
                if session.hasGPS {
                    badge("GPS", color: .green)
                }
                if session.hasBLE {
                    badge("BLE", color: .orange)
                }
                badge(session.startMethod.rawValue, color: .purple)
            }
            .padding(.top, 2)

            if let size = session.totalFileSizeBytes {
                Text(formatBytes(size))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(.caption2, design: .monospaced))
            .fontWeight(.medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundColor(color)
            .cornerRadius(4)
    }

    private func deleteSessions(at offsets: IndexSet) {
        for index in offsets {
            SessionStore.shared.deleteSession(sessions[index])
        }
        sessions = SessionStore.shared.loadSessions()
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
