import Foundation

enum SenseVoiceFrontendError: LocalizedError {
    case emptyInput
    case invalidSampleRate
    case featureExtractionFailed(String)

    var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "音频输入为空"
        case .invalidSampleRate:
            return "采样率无效"
        case .featureExtractionFailed(let message):
            return message
        }
    }
}

final class SenseVoiceFrontend {
    let sampleRate: Int
    let nMels: Int
    let frameLengthMs: Float
    let frameShiftMs: Float
    let nFFT: Int
    let lowFreq: Float
    let highFreq: Float
    let lfrM: Int
    let lfrN: Int
    let eps: Float

    private let processor: FastFbankProcessor

    init(
        sampleRate: Int = 16000,
        nMels: Int = 80,
        frameLengthMs: Float = 25.0,
        frameShiftMs: Float = 10.0,
        nFFT: Int = 512,
        lowFreq: Float = 0.0,
        highFreq: Float? = nil,
        lfrM: Int = 7,
        lfrN: Int = 6,
        eps: Float = 1e-10
    ) {
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.frameLengthMs = frameLengthMs
        self.frameShiftMs = frameShiftMs
        self.nFFT = nFFT
        self.lowFreq = lowFreq
        self.highFreq = highFreq ?? Float(sampleRate) / 2.0
        self.lfrM = lfrM
        self.lfrN = lfrN
        self.eps = eps

        self.processor = FastFbankProcessor(
            sampleRate: Float(sampleRate),
            numMelBins: nMels,
            frameLengthMs: frameLengthMs,
            frameShiftMs: frameShiftMs,
            lowFreq: lowFreq,
            highFreq: highFreq ?? Float(sampleRate) / 2.0,
            fftSize: nFFT,
            useLogFbank: true,
            eps: eps
        )!
    }

    func extractFeatures(from samples: [Float]) throws -> [[Float]] {
        guard sampleRate > 0 else {
            throw SenseVoiceFrontendError.invalidSampleRate
        }
        guard !samples.isEmpty else {
            throw SenseVoiceFrontendError.emptyInput
        }

        let logMel = processor.compute(waveform: samples)
        let lfr = applyLFR(logMel, m: lfrM, n: lfrN)

        if lfr.isEmpty || lfr[0].isEmpty {
            throw SenseVoiceFrontendError.featureExtractionFailed("LFR 处理失败")
        }

        return lfr
    }

    private func applyLFR(_ features: [[Float]], m: Int, n: Int) -> [[Float]] {
        guard !features.isEmpty else { return [] }

        let dimension = features[0].count
        let leftPadCount = (m - 1) / 2
        let leftPadding = Array(repeating: features[0], count: leftPadCount)
        let padded = leftPadding + features
        let totalFrames = padded.count
        let lfrFrameCount = Int(ceil(Double(totalFrames - m) / Double(n))) + 1

        var result: [[Float]] = []
        result.reserveCapacity(max(0, lfrFrameCount))

        for index in 0..<lfrFrameCount {
            let start = index * n
            let end = start + m

            var stacked: [Float] = []
            stacked.reserveCapacity(m * dimension)

            if end <= totalFrames {
                for frame in start..<end {
                    stacked.append(contentsOf: padded[frame])
                }
            } else {
                for frame in start..<totalFrames {
                    stacked.append(contentsOf: padded[frame])
                }
                let padCount = end - totalFrames
                for _ in 0..<padCount {
                    stacked.append(contentsOf: padded[totalFrames - 1])
                }
            }

            result.append(stacked)
        }

        return result
    }
}
