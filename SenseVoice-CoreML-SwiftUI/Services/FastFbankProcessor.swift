import Accelerate
import Foundation

@inline(__always)
private func nextPowerOf2(_ n: Int) -> Int {
    if n <= 1 { return 1 }
    var value = n - 1
    value |= value >> 1
    value |= value >> 2
    value |= value >> 4
    value |= value >> 8
    value |= value >> 16
    return value + 1
}

@inline(__always)
private func hzToMel(_ hz: Float) -> Float {
    1127.0 * log(1.0 + hz / 700.0)
}

private func makePoveyWindow(length: Int) -> [Float] {
    precondition(length >= 2)
    var window = [Float](repeating: 0, count: length)
    for index in 0..<length {
        let factor = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(index) / Float(length - 1))
        window[index] = pow(factor, 0.85)
    }
    return window
}

private func makeMelFilterbankFlat(
    sampleRate: Float,
    fftSize: Int,
    numMelBins: Int,
    lowFreq: Float,
    highFreq: Float
) -> [Float] {
    let numFftBins = fftSize / 2 + 1
    let lowMel = hzToMel(lowFreq)
    let highMel = hzToMel(highFreq)

    var centerMels = [Float](repeating: 0, count: numMelBins + 2)
    for index in 0..<(numMelBins + 2) {
        centerMels[index] = lowMel + Float(index) * (highMel - lowMel) / Float(numMelBins + 1)
    }

    var filterBank = [Float](repeating: 0, count: numMelBins * numFftBins)
    for fftBin in 0..<numFftBins {
        let mel = hzToMel(Float(fftBin) * sampleRate / Float(fftSize))
        for melIndex in 0..<numMelBins {
            let left = centerMels[melIndex]
            let center = centerMels[melIndex + 1]
            let right = centerMels[melIndex + 2]

            let weight: Float
            if mel > left && mel < center {
                weight = (mel - left) / (center - left)
            } else if mel >= center && mel < right {
                weight = (right - mel) / (right - center)
            } else {
                weight = 0
            }

            filterBank[melIndex * numFftBins + fftBin] = weight
        }
    }

    return filterBank
}

private final class FFTPlan {
    let fftSize: Int
    let halfSize: Int
    let log2n: vDSP_Length
    let setup: FFTSetup

    private var real: [Float]
    private var imag: [Float]
    private var interleaved: [DSPComplex]

    init?(fftSize: Int) {
        guard fftSize > 0, (fftSize & (fftSize - 1)) == 0 else { return nil }

        self.fftSize = fftSize
        self.halfSize = fftSize / 2
        self.log2n = vDSP_Length(log2(Float(fftSize)))

        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return nil
        }

        self.setup = setup
        self.real = [Float](repeating: 0, count: fftSize / 2)
        self.imag = [Float](repeating: 0, count: fftSize / 2)
        self.interleaved = [DSPComplex](repeating: DSPComplex(), count: fftSize / 2)
    }

    deinit {
        vDSP_destroy_fftsetup(setup)
    }

    func powerSpectrum(realInput: [Float], out: inout [Float]) {
        precondition(realInput.count == fftSize)
        precondition(out.count == halfSize + 1)

        for index in 0..<halfSize {
            interleaved[index].real = realInput[2 * index]
            interleaved[index].imag = realInput[2 * index + 1]
        }

        real.withUnsafeMutableBufferPointer { realBuffer in
            imag.withUnsafeMutableBufferPointer { imagBuffer in
                var split = DSPSplitComplex(
                    realp: realBuffer.baseAddress!,
                    imagp: imagBuffer.baseAddress!
                )

                interleaved.withUnsafeMutableBufferPointer { interleavedBuffer in
                    vDSP_ctoz(interleavedBuffer.baseAddress!, 2, &split, 1, vDSP_Length(halfSize))
                }

                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))

                out[0] = split.realp[0] * split.realp[0]
                out[halfSize] = split.imagp[0] * split.imagp[0]

                out.withUnsafeMutableBufferPointer { outputBuffer in
                    vDSP_zvmags(&split, 1, outputBuffer.baseAddress! + 1, 1, vDSP_Length(halfSize - 1))
                }

                var scale = Float(1.0 / Float(fftSize))
                vDSP_vsmul(out, 1, &scale, &out, 1, vDSP_Length(halfSize + 1))
            }
        }
    }
}

final class FastFbankProcessor {
    let sampleRate: Float
    let numMelBins: Int
    let frameLength: Int
    let frameShift: Int
    let fftSize: Int
    let numFftBins: Int
    let useLogFbank: Bool
    let eps: Float

    let window: [Float]
    let melFB: [Float]
    private let fftPlan: FFTPlan

    private var frameBuffer: [Float]
    private var paddedBuffer: [Float]
    private var powerBuffer: [Float]
    private var melBuffer: [Float]

    init?(
        sampleRate: Float,
        numMelBins: Int,
        frameLengthMs: Float,
        frameShiftMs: Float,
        lowFreq: Float,
        highFreq: Float,
        fftSize: Int,
        useLogFbank: Bool,
        eps: Float
    ) {
        self.sampleRate = sampleRate
        self.numMelBins = numMelBins
        self.frameLength = Int(sampleRate * frameLengthMs / 1000.0)
        self.frameShift = Int(sampleRate * frameShiftMs / 1000.0)
        self.fftSize = max(fftSize, nextPowerOf2(self.frameLength))
        self.numFftBins = self.fftSize / 2 + 1
        self.useLogFbank = useLogFbank
        self.eps = eps

        self.window = makePoveyWindow(length: self.frameLength)
        self.melFB = makeMelFilterbankFlat(
            sampleRate: sampleRate,
            fftSize: self.fftSize,
            numMelBins: numMelBins,
            lowFreq: lowFreq,
            highFreq: highFreq
        )

        guard let fftPlan = FFTPlan(fftSize: self.fftSize) else {
            return nil
        }

        self.fftPlan = fftPlan
        self.frameBuffer = [Float](repeating: 0, count: self.frameLength)
        self.paddedBuffer = [Float](repeating: 0, count: self.fftSize)
        self.powerBuffer = [Float](repeating: 0, count: self.numFftBins)
        self.melBuffer = [Float](repeating: 0, count: numMelBins)
    }

    func compute(waveform: [Float]) -> [[Float]] {
        guard !waveform.isEmpty else { return [] }

        let frameCount: Int
        if waveform.count <= frameLength {
            frameCount = 1
        } else {
            frameCount = Int(ceil(Float(waveform.count - frameLength) / Float(frameShift))) + 1
        }

        var result: [[Float]] = []
        result.reserveCapacity(frameCount)

        for frameIndex in 0..<frameCount {
            let start = frameIndex * frameShift
            let available = max(0, min(frameLength, waveform.count - start))

            frameBuffer.withUnsafeMutableBufferPointer { buffer in
                buffer.initialize(repeating: 0)
            }

            if available > 0 {
                frameBuffer.withUnsafeMutableBufferPointer { dst in
                    waveform.withUnsafeBufferPointer { src in
                        dst.baseAddress!.update(from: src.baseAddress! + start, count: available)
                    }
                }
            }

            vDSP_vmul(frameBuffer, 1, window, 1, &frameBuffer, 1, vDSP_Length(frameLength))

            paddedBuffer.withUnsafeMutableBufferPointer { buffer in
                buffer.initialize(repeating: 0)
            }

            paddedBuffer.withUnsafeMutableBufferPointer { dst in
                frameBuffer.withUnsafeBufferPointer { src in
                    dst.baseAddress!.update(from: src.baseAddress!, count: frameLength)
                }
            }

            fftPlan.powerSpectrum(realInput: paddedBuffer, out: &powerBuffer)

            melFB.withUnsafeBufferPointer { melPtr in
                powerBuffer.withUnsafeBufferPointer { powerPtr in
                    melBuffer.withUnsafeMutableBufferPointer { outPtr in
                        vDSP_mmul(
                            melPtr.baseAddress!, 1,
                            powerPtr.baseAddress!, 1,
                            outPtr.baseAddress!, 1,
                            vDSP_Length(numMelBins),
                            1,
                            vDSP_Length(numFftBins)
                        )
                    }
                }
            }

            if useLogFbank {
                var low = eps
                var high = Float.greatestFiniteMagnitude
                vDSP_vclip(melBuffer, 1, &low, &high, &melBuffer, 1, vDSP_Length(numMelBins))

                var count = Int32(numMelBins)
                vvlogf(&melBuffer, melBuffer, &count)
            }

            result.append(melBuffer)
        }

        return result
    }
}
