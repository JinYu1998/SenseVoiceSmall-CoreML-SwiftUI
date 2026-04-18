import AVFoundation
import Combine
import Foundation

final class AudioRecorder: NSObject, ObservableObject, AVAudioRecorderDelegate {
    private var recorder: AVAudioRecorder?
    private var timer: Timer?

    @Published var duration: TimeInterval = 0
    private(set) var audioURL: URL?

    func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    func startRecording() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("record.wav")
        if FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.removeItem(at: url)
        }

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]

        recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder?.delegate = self
        recorder?.isMeteringEnabled = false
        recorder?.prepareToRecord()

        guard recorder?.record() == true else {
            throw SenseVoiceError.audioDecodeFailed("录音启动失败")
        }

        audioURL = url
        duration = 0

        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] timer in
            guard let self, let recorder = self.recorder, recorder.isRecording else {
                timer.invalidate()
                return
            }

            Task { @MainActor in
                self.duration = recorder.currentTime
            }
        }
    }

    func stopRecording() {
        recorder?.stop()
        timer?.invalidate()
        timer = nil
        duration = recorder?.currentTime ?? duration
    }
}
