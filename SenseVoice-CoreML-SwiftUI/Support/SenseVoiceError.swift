import Foundation

enum SenseVoiceError: LocalizedError {
    case resourceNotFound(String)
    case audioDecodeFailed(String)
    case featureExtractionFailed(String)
    case modelOutputMissing(String)

    var errorDescription: String? {
        switch self {
        case .resourceNotFound(let message),
             .audioDecodeFailed(let message),
             .featureExtractionFailed(let message),
             .modelOutputMissing(let message):
            return message
        }
    }
}
