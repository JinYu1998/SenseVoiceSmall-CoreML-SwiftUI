import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = SenseVoiceViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                actionButtons
                recordingInfoSection
                rawTextSection
                finalTextSection
            }
            .padding()
            .navigationTitle("SenseVoice")
        }
    }

    private var actionButtons: some View {
        VStack(spacing: 12) {
            Button {
                Task {
                    await viewModel.transcribeBundledAudio()
                }
            } label: {
                HStack {
                    if viewModel.isRunning && !viewModel.isRecording {
                        ProgressView()
                            .progressViewStyle(.circular)
                    }

                    Text(viewModel.isRunning && !viewModel.isRecording ? "识别中..." : "开始识别 test.wav")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue.opacity(0.15))
                .cornerRadius(12)
            }
            .disabled(viewModel.isRunning || viewModel.isRecording)

            Button {
                Task {
                    await viewModel.toggleRecording()
                }
            } label: {
                HStack {
                    if viewModel.isRecording {
                        ProgressView()
                            .progressViewStyle(.circular)
                    }

                    Text(viewModel.isRecording ? "停止录音" : "🎤 开始录音")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.red.opacity(0.15))
                .cornerRadius(12)
            }
            .disabled(viewModel.isRunning && !viewModel.isRecording)
        }
    }

    private var recordingInfoSection: some View {
        GroupBox("录音信息") {
            VStack(alignment: .leading, spacing: 8) {
                Text("录音状态：\(viewModel.isRecording ? "录音中" : "未录音")")
                Text("录音时长：\(viewModel.recordingDurationText) 秒")
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var rawTextSection: some View {
        GroupBox("Raw Text") {
            ScrollView {
                Text(viewModel.rawText.isEmpty ? "暂无" : viewModel.rawText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(maxHeight: 180)
        }
    }

    private var finalTextSection: some View {
        GroupBox("Final Text") {
            ScrollView {
                Text(viewModel.resultText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(maxHeight: .infinity)
        }
    }
}
