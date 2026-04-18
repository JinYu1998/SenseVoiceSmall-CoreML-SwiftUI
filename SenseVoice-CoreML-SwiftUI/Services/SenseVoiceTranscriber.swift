import AVFoundation
import Accelerate
import CoreML
import Foundation

final class SenseVoiceTranscriber {
    static let shared: SenseVoiceTranscriber = {
        do {
            return try SenseVoiceTranscriber()
        } catch {
            fatalError("SenseVoiceTranscriber 初始化失败: \(error.localizedDescription)")
        }
    }()

    private let blankID = 0
    private let fixedFrames = 300
    private let featureDim = 560

    private let frontend = SenseVoiceFrontend(
        sampleRate: 16000,
        nMels: 80,
        frameLengthMs: 25.0,
        frameShiftMs: 10.0,
        nFFT: 512,
        lowFreq: 0.0,
        highFreq: nil,
        lfrM: 7,
        lfrN: 6,
        eps: 1e-10
    )

    private let model: MLModel
    private let tokens: [String]

    private init() throws {
        guard let modelURL = Bundle.main.url(
            forResource: "SenseVoiceSmall300",
            withExtension: "mlmodelc"
        ) else {
            throw SenseVoiceError.resourceNotFound("找不到 model/SenseVoiceSmall300.mlmodelc。首次使用请先执行 scripts/download_model.sh，把模型下载到 Model/ 目录。")
        }

        guard let tokenURL = Bundle.main.url(forResource: "tokens", withExtension: "txt") else {
            throw SenseVoiceError.resourceNotFound("找不到 tokens.txt")
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        model = try MLModel(contentsOf: modelURL, configuration: config)
        tokens = try Self.loadTokens(from: tokenURL)
    }

    func transcribe(audioURL: URL) throws -> SenseVoiceOutput {
        let startTotal = Date()

        let startStep2 = Date()
        let floatSamples = try loadWavAsFloatArray(url: audioURL, targetSampleRate: 16000)
        guard !floatSamples.isEmpty else {
            throw SenseVoiceError.audioDecodeFailed("音频为空")
        }
        print("Step 2: 读取音频 + 转 Float     \(Date().timeIntervalSince(startStep2)) s")

        let startStep3 = Date()
        let lfr = try frontend.extractFeatures(from: floatSamples)
        if lfr.isEmpty || lfr[0].isEmpty {
            throw SenseVoiceError.featureExtractionFailed("fbank/LFR 提取失败")
        }
        print("Step 3: 特征提取 (fbank/LFR)   \(Date().timeIntervalSince(startStep3)) s")

        let startStep4 = Date()
        let fixedLFR = padOrTrimFeatures(lfr, to: fixedFrames, featureDim: featureDim)
        print("Step 4: 固定长度处理           \(Date().timeIntervalSince(startStep4)) s")

        let startStep5 = Date()
        let speechArray = try make3DMLMultiArray(from: fixedLFR)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["speech": speechArray])
        let output = try model.prediction(from: provider)
        print("Step 5: CoreML 推理            \(Date().timeIntervalSince(startStep5)) s")

        guard let ctcLogits = output.featureValue(for: "ctc_logits")?.multiArrayValue else {
            throw SenseVoiceError.modelOutputMissing("缺少输出 ctc_logits")
        }
        guard let encoderOutLens = output.featureValue(for: "encoder_out_lens")?.multiArrayValue else {
            throw SenseVoiceError.modelOutputMissing("缺少输出 encoder_out_lens")
        }

        let startStep6 = Date()
        let rawText = try greedyDecode(
            ctcLogits: ctcLogits,
            encoderOutLens: encoderOutLens,
            tokens: tokens,
            blankID: blankID
        )
        print("Step 6: CTC greedy decode      \(Date().timeIntervalSince(startStep6)) s")

        let startStep7 = Date()
        let finalText = postprocessSenseVoiceText(rawText)
        print("Step 7: 后处理                 \(Date().timeIntervalSince(startStep7)) s")

        print("Total: 总耗时                  \(Date().timeIntervalSince(startTotal)) s")

        return SenseVoiceOutput(rawText: rawText, finalText: finalText)
    }
}

private extension SenseVoiceTranscriber {
    func loadWavAsFloatArray(url: URL, targetSampleRate: Double) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw SenseVoiceError.audioDecodeFailed("无法创建目标音频格式")
        }

        let inputFrameCount = AVAudioFrameCount(file.length)
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: inputFrameCount
        ) else {
            throw SenseVoiceError.audioDecodeFailed("无法创建输入 buffer")
        }

        try file.read(into: inputBuffer)

        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw SenseVoiceError.audioDecodeFailed("无法创建 AVAudioConverter")
        }

        let ratio = targetSampleRate / sourceFormat.sampleRate
        let estimatedFrameCapacity = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio + 1024)

        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: estimatedFrameCapacity
        ) else {
            throw SenseVoiceError.audioDecodeFailed("无法创建输出 buffer")
        }

        var didProvideInput = false
        var conversionError: NSError?

        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if didProvideInput {
                outStatus.pointee = .endOfStream
                return nil
            }

            didProvideInput = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if let conversionError {
            throw conversionError
        }

        guard status == .haveData || status == .endOfStream || status == .inputRanDry else {
            throw SenseVoiceError.audioDecodeFailed("音频转换失败，status=\(status.rawValue)")
        }

        guard outputBuffer.frameLength > 0 else {
            throw SenseVoiceError.audioDecodeFailed("转换后没有输出音频帧")
        }

        guard let channelData = outputBuffer.int16ChannelData else {
            throw SenseVoiceError.audioDecodeFailed("拿不到 Int16 声道数据")
        }

        let frames = Int(outputBuffer.frameLength)
        let src = channelData[0]

        var result = [Float](repeating: 0, count: frames)
        vDSP_vflt16(src, 1, &result, 1, vDSP_Length(frames))

        var scale: Float = 1.0 / 32768.0
        vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(frames))
        return result
    }

    func padOrTrimFeatures(_ input: [[Float]], to targetT: Int, featureDim: Int) -> [[Float]] {
        guard targetT > 0 else { return [] }

        if input.isEmpty {
            return Array(repeating: Array(repeating: 0, count: featureDim), count: targetT)
        }

        if input.count == targetT {
            return input
        }

        if input.count > targetT {
            return Array(input.prefix(targetT))
        }

        let zeroFrame = Array(repeating: Float(0), count: featureDim)
        return input + Array(repeating: zeroFrame, count: targetT - input.count)
    }

    func make3DMLMultiArray(from data: [[Float]]) throws -> MLMultiArray {
        guard !data.isEmpty else {
            throw SenseVoiceError.featureExtractionFailed("LFR 输出为空")
        }

        let t = data.count
        let d = data[0].count

        let array = try MLMultiArray(
            shape: [1, NSNumber(value: t), NSNumber(value: d)],
            dataType: .float32
        )

        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: t * d)
        var offset = 0

        for row in data {
            row.withUnsafeBufferPointer { rowPtr in
                ptr.advanced(by: offset).assign(from: rowPtr.baseAddress!, count: d)
            }
            offset += d
        }

        return array
    }

    static func loadTokens(from url: URL) throws -> [String] {
        let content = try String(contentsOf: url, encoding: .utf8)
        return content
            .split(whereSeparator: \.isNewline)
            .compactMap { line in
                line.split(whereSeparator: \.isWhitespace).first.map(String.init)
            }
            .filter { !$0.isEmpty }
    }

    func greedyDecode(
        ctcLogits: MLMultiArray,
        encoderOutLens: MLMultiArray,
        tokens: [String],
        blankID: Int
    ) throws -> String {
        let shape = ctcLogits.shape.map(\.intValue)
        guard shape.count == 3 else {
            throw SenseVoiceError.modelOutputMissing("ctc_logits 维度不是 3D")
        }

        let batch = shape[0]
        let time = shape[1]
        let vocab = shape[2]

        guard batch >= 1 else {
            throw SenseVoiceError.modelOutputMissing("ctc_logits batch 为空")
        }

        let validLen = max(0, min(readFirstInt(from: encoderOutLens), time))
        let strides = ctcLogits.strides.map(\.intValue)

        guard strides.count == 3 else {
            throw SenseVoiceError.modelOutputMissing("ctc_logits strides 非法")
        }

        let batchStride = strides[0]
        let timeStride = strides[1]
        let vocabStride = strides[2]

        var predIDs: [Int] = []
        predIDs.reserveCapacity(validLen)
        let batchBase = 0 * batchStride

        switch ctcLogits.dataType {
        case .float32:
            let ptr = ctcLogits.dataPointer.bindMemory(to: Float32.self, capacity: ctcLogits.count)
            if vocabStride == 1 {
                for t in 0..<validLen {
                    let timeBase = batchBase + t * timeStride
                    var maxValue: Float = 0
                    var maxIndex: vDSP_Length = 0
                    vDSP_maxvi(ptr.advanced(by: timeBase), 1, &maxValue, &maxIndex, vDSP_Length(vocab))
                    predIDs.append(Int(maxIndex))
                }
            } else {
                for t in 0..<validLen {
                    var maxValue = -Float.greatestFiniteMagnitude
                    var maxIndex = 0
                    let timeBase = batchBase + t * timeStride
                    for v in 0..<vocab {
                        let value = ptr[timeBase + v * vocabStride]
                        if value > maxValue {
                            maxValue = value
                            maxIndex = v
                        }
                    }
                    predIDs.append(maxIndex)
                }
            }

        case .float16:
            let ptr = ctcLogits.dataPointer.bindMemory(to: Float16.self, capacity: ctcLogits.count)
            for t in 0..<validLen {
                var maxValue = -Float16.infinity
                var maxIndex = 0
                let timeBase = batchBase + t * timeStride
                for v in 0..<vocab {
                    let value = ptr[timeBase + v * vocabStride]
                    if value > maxValue {
                        maxValue = value
                        maxIndex = v
                    }
                }
                predIDs.append(maxIndex)
            }

        case .double:
            let ptr = ctcLogits.dataPointer.bindMemory(to: Double.self, capacity: ctcLogits.count)
            if vocabStride == 1 {
                for t in 0..<validLen {
                    let timeBase = batchBase + t * timeStride
                    var maxValue: Double = 0
                    var maxIndex: vDSP_Length = 0
                    vDSP_maxviD(ptr.advanced(by: timeBase), 1, &maxValue, &maxIndex, vDSP_Length(vocab))
                    predIDs.append(Int(maxIndex))
                }
            } else {
                for t in 0..<validLen {
                    var maxValue = -Double.greatestFiniteMagnitude
                    var maxIndex = 0
                    let timeBase = batchBase + t * timeStride
                    for v in 0..<vocab {
                        let value = ptr[timeBase + v * vocabStride]
                        if value > maxValue {
                            maxValue = value
                            maxIndex = v
                        }
                    }
                    predIDs.append(maxIndex)
                }
            }

        default:
            for t in 0..<validLen {
                var maxValue = -Float.greatestFiniteMagnitude
                var maxIndex = 0
                for v in 0..<vocab {
                    let value = ctcLogits[[0, t as NSNumber, v as NSNumber]].floatValue
                    if value > maxValue {
                        maxValue = value
                        maxIndex = v
                    }
                }
                predIDs.append(maxIndex)
            }
        }

        var decodedIDs: [Int] = []
        decodedIDs.reserveCapacity(predIDs.count)

        var prev = -1
        for idx in predIDs {
            if idx != prev && idx != blankID {
                decodedIDs.append(idx)
            }
            prev = idx
        }

        let skipSet: Set<String> = ["<blank>", "<blk>", "<s>", "</s>", "<unk>", "<pad>"]
        var pieces: [String] = []
        pieces.reserveCapacity(decodedIDs.count)

        for idx in decodedIDs where idx >= 0 && idx < tokens.count {
            let token = tokens[idx]
            if skipSet.contains(token) {
                continue
            }
            pieces.append(token)
        }

        return pieces.joined()
    }

    func readFirstInt(from array: MLMultiArray) -> Int {
        array.count > 0 ? array[0].intValue : 0
    }

    func postprocessSenseVoiceText(_ text: String) -> String {
        var result = text

        let tagsToRemove = [
            "<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>",
            "<|nospeech|>", "<|Event_UNK|>", "<|NEUTRAL|>",
            "<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|SURPRISED|>",
            "<|BGM|>", "<|Speech|>", "<|Applause|>", "<|Laughter|>",
            "<|Cry|>", "<|Sneeze|>", "<|Breath|>", "<|Cough|>",
            "<|Sing|>", "<|Speech_Noise|>", "<|withitn|>", "<|woitn|>"
        ]

        for tag in tagsToRemove {
            result = result.replacingOccurrences(of: tag, with: "")
        }

        result = result.replacingOccurrences(of: "▁", with: " ")
        result = result.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
