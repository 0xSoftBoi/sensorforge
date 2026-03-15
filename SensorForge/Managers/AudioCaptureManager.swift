import Foundation
import AVFoundation
import Combine

/// Captures spatial audio (multi-channel) via AVAudioEngine and writes to WAV files.
final class AudioCaptureManager: ObservableObject {
    @Published var isRunning = false
    @Published var currentLevel: Float = 0

    private var audioEngine: AVAudioEngine?
    private var audioFile: AVAudioFile?
    private var dataStore: SensorDataStore?
    private var sessionDirectory: URL?
    private var startTime: TimeInterval = 0

    func start(dataStore: SensorDataStore, sessionDirectory: URL) {
        self.dataStore = dataStore
        self.sessionDirectory = sessionDirectory

        configureAudioSession()
        startEngine(sessionDirectory: sessionDirectory)
    }

    func stop() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil

        if let audioFile, let sessionDirectory {
            let duration = ProcessInfo.processInfo.systemUptime - startTime
            let metadata = AudioChunkMetadata(
                timestamp: SensorTimestamp(bootTime: startTime),
                sampleRate: audioFile.processingFormat.sampleRate,
                channelCount: Int(audioFile.processingFormat.channelCount),
                durationSeconds: duration,
                filePath: audioFile.url.lastPathComponent
            )
            Task { @MainActor in
                self.dataStore?.audioChunks.append(metadata)
            }
        }

        audioFile = nil
        isRunning = false
    }

    private func configureAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: [])
            try audioSession.setPreferredSampleRate(48000)
            try audioSession.setActive(true)
        } catch {
            print("[AudioCapture] Failed to configure audio session: \(error)")
        }
    }

    private func startEngine(sessionDirectory: URL) {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Create output file
        let audioURL = sessionDirectory.appendingPathComponent("spatial_audio.wav")
        do {
            audioFile = try AVAudioFile(forWriting: audioURL, settings: inputFormat.settings)
        } catch {
            print("[AudioCapture] Failed to create audio file: \(error)")
            return
        }

        // Install tap for recording and level metering
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, time in
            guard let self else { return }

            // Write to file
            try? self.audioFile?.write(from: buffer)

            // Update level meter
            let level = self.calculateRMSLevel(buffer: buffer)
            DispatchQueue.main.async {
                self.currentLevel = level
            }
        }

        do {
            try engine.start()
            self.audioEngine = engine
            self.startTime = ProcessInfo.processInfo.systemUptime
            isRunning = true
        } catch {
            print("[AudioCapture] Failed to start engine: \(error)")
        }
    }

    private func calculateRMSLevel(buffer: AVAudioPCMBuffer) -> Float {
        guard let channelData = buffer.floatChannelData else { return 0 }
        let channelCount = Int(buffer.format.channelCount)
        let frameLength = Int(buffer.frameLength)
        guard frameLength > 0 else { return 0 }

        var rms: Float = 0
        for channel in 0..<channelCount {
            let data = channelData[channel]
            var sum: Float = 0
            for frame in 0..<frameLength {
                let sample = data[frame]
                sum += sample * sample
            }
            rms += sqrt(sum / Float(frameLength))
        }
        return rms / Float(channelCount)
    }
}
