import Combine
import Foundation

@MainActor
final class SenseVoiceViewModel: ObservableObject {
    @Published var resultText: String = "点击按钮开始识别"
    @Published var rawText: String = ""
    @Published var isRunning: Bool = false
    @Published var isRecording: Bool = false
    @Published private(set) var recordDuration: Double = 0

    let recorder: AudioRecorder
    private let transcriber: SenseVoiceTranscriber
    private var cancellables: Set<AnyCancellable> = []

    init(
        recorder: AudioRecorder = AudioRecorder(),
        transcriber: SenseVoiceTranscriber = .shared
    ) {
        self.recorder = recorder
        self.transcriber = transcriber

        recorder.objectWillChange
            .sink { [weak self] _ in
                self?.objectWillChange.send()
            }
            .store(in: &cancellables)
    }

    var recordingDurationText: String {
        let duration = isRecording ? recorder.duration : recordDuration
        return String(format: "%.2f", duration)
    }

    func transcribeBundledAudio() async {
        isRunning = true
        rawText = ""
        resultText = "处理中..."
        recordDuration = 0

        defer { isRunning = false }

        do {
            guard let wavURL = Bundle.main.url(forResource: "test", withExtension: "wav") else {
                throw SenseVoiceError.resourceNotFound("找不到 test.wav")
            }

            let output = try transcriber.transcribe(audioURL: wavURL)
            rawText = output.rawText
            resultText = output.finalText
        } catch {
            resultText = "识别失败：\(error.localizedDescription)"
        }
    }

    func toggleRecording() async {
        if isRecording {
            recorder.stopRecording()
            isRecording = false
            recordDuration = recorder.duration

            guard let url = recorder.audioURL else {
                resultText = "录音失败：未找到录音文件"
                return
            }

            await transcribeRecording(at: url)
            return
        }

        do {
            let granted = await recorder.requestPermission()
            guard granted else {
                resultText = "麦克风权限未开启"
                return
            }

            try recorder.startRecording()
            rawText = ""
            resultText = "录音中，请再次点击按钮结束录音..."
            recordDuration = 0
            isRecording = true
        } catch {
            resultText = "无法启动录音：\(error.localizedDescription)"
        }
    }

    private func transcribeRecording(at url: URL) async {
        isRunning = true
        rawText = ""
        resultText = "处理中..."

        defer { isRunning = false }

        do {
            let output = try transcriber.transcribe(audioURL: url)
            rawText = output.rawText
            resultText = """
            录音时长: \(String(format: "%.2f", recordDuration)) 秒

            \(output.finalText)
            """
        } catch {
            resultText = "识别失败：\(error.localizedDescription)"
        }
    }
}
