# SenseVoice-CoreML-SwiftUI

一个基于 SwiftUI 的本地语音识别 Demo，使用 Core ML 版本的 SenseVoice 模型完成音频前处理、推理和文本后处理。当前项目支持两种入口：

- 识别打包进 App 的 `test.wav`
- 使用麦克风录音后立即转写

## 模型来源与转换说明

当前项目固定使用 `SenseVoiceSmall300` 这个 Core ML 模型。

- 模型转换参考项目：[`mefengl/SenseVoiceSmall-coreml`](https://github.com/mefengl/SenseVoiceSmall-coreml)
- 我当前项目里使用的转换流程，是在它的 `convert_coreml` 基础上做过改动的版本
- 具体改动细节我后续会再补充到这个 README

也就是说，这个仓库里的 Core ML 模型不是直接原样使用上游仓库的转换产物，而是基于它的思路和脚本进一步调整后的结果。

## 首次使用前先下载模型

因为 `SenseVoiceSmall300` 模型体积较大，不适合直接放进 git 仓库，所以模型文件托管在 Hugging Face：

- 模型地址：[`Mor1998/SenseVoiceSmall300`](https://huggingface.co/Mor1998/SenseVoiceSmall300/tree/main)

首次拉取这个仓库后，请先把模型下载到本地 `Model/` 目录，再打开 Xcode 编译运行。

推荐方式：

```bash
sh scripts/download_model.sh
```

这个脚本会尝试把 Hugging Face 仓库 `Mor1998/SenseVoiceSmall300` 的内容下载到当前项目的 `Model/` 文件夹下。
如果 Hugging Face 仓库里保存的是 `.mlmodelc` 包内部文件，脚本也会自动把它们整理成：

```bash
Model/SenseVoiceSmall300.mlmodelc
```

如果你的本机还没有安装 Hugging Face CLI，可以先安装：

```bash
pip install -U "huggingface_hub[cli]"
```

然后再执行：

```bash
hf download Mor1998/SenseVoiceSmall300 --local-dir Model
```

下载完成后，项目里应当至少能在 `Model/` 下看到 `SenseVoiceSmall300.mlmodelc`。

## 当前代码结构

- `SenseVoice-CoreML-SwiftUI/ContentView.swift`
  只负责页面布局和触发用户操作。
- `SenseVoice-CoreML-SwiftUI/ViewModels/SenseVoiceViewModel.swift`
  负责 UI 状态、录音流程切换、调用转写服务。
- `SenseVoice-CoreML-SwiftUI/Services/AudioRecorder.swift`
  负责麦克风权限申请、录音文件生成、录音时长更新。
- `SenseVoice-CoreML-SwiftUI/Services/SenseVoiceTranscriber.swift`
  负责模型加载、音频读取、特征转张量、Core ML 推理、CTC greedy decode、文本后处理。
- `SenseVoice-CoreML-SwiftUI/Services/SenseVoiceFrontend.swift`
  负责 log-mel fbank 和 LFR 特征拼接。
- `SenseVoice-CoreML-SwiftUI/Services/FastFbankProcessor.swift`
  负责 FFT、Mel Filterbank 和 fbank 底层计算。
- `SenseVoice-CoreML-SwiftUI/Models/SenseVoiceOutput.swift`
  负责转写结果数据结构。
- `SenseVoice-CoreML-SwiftUI/Support/SenseVoiceError.swift`
  统一定义项目内错误类型。

## 当前处理方式

应用当前的处理链路如下：

1. 用户点击按钮，选择识别 `test.wav` 或开始录音。
2. 如果是录音流程，先申请麦克风权限，再用 `AVAudioRecorder` 生成 16k 单声道 WAV。
3. `SenseVoiceTranscriber` 读取音频，并转换为模型需要的浮点采样。
4. `SenseVoiceFrontend` 调用 `FastFbankProcessor` 生成 log-mel fbank，再做 LFR 堆叠。
5. 特征被裁剪或补齐到固定帧长后，转成 `MLMultiArray`。
6. Core ML 模型输出 `ctc_logits` 与 `encoder_out_lens`。
7. 通过 greedy decode 生成原始 token 串，再做标签清理、空白压缩与文本整理。
8. 最终结果回写到 SwiftUI 页面。

## SenseVoiceSmall300 模型音频处理细节

当前项目固定使用 `Model/SenseVoiceSmall300.mlmodelc`，所以整条音频处理链路也是按这个模型的输入要求写死的，没有做动态切换。

### 1. 输入音频规格

- 目标采样率固定为 `16000 Hz`
- 单声道 `1 channel`
- PCM `Int16`
- 录音时直接按这个格式写入 `record.wav`
- 外部 WAV 文件读入后，如果原始采样率或声道不一致，会先通过 `AVAudioConverter` 转成目标格式

### 2. 波形归一化

音频在读入后，会从 `Int16` 声道数据转成 `Float`，并按下面的方式归一化：

- 每个采样点除以 `32768.0`
- 输出范围约为 `[-1, 1]`

这一步得到的是后续特征提取使用的原始浮点波形。

### 3. Fbank 参数

`SenseVoiceFrontend` 当前固定参数如下：

- `sampleRate = 16000`
- `nMels = 80`
- `frameLengthMs = 25.0`
- `frameShiftMs = 10.0`
- `nFFT = 512`
- `lowFreq = 0.0`
- `highFreq = sampleRate / 2 = 8000`
- `eps = 1e-10`

对应到实际含义：

- 每帧窗长约为 `25 ms`，也就是 `400` 个采样点
- 帧移约为 `10 ms`，也就是 `160` 个采样点
- 每帧先做加窗、FFT 和功率谱计算
- 再通过 `80` 维 Mel Filterbank 得到 log-mel fbank 特征

当前 `FastFbankProcessor` 里使用的是 Povey window，并在 Mel 能量上取对数。

### 4. LFR 拼帧

为了匹配 `SenseVoiceSmall300` 的输入，项目在 log-mel fbank 后又做了一层 LFR：

- `lfrM = 7`
- `lfrN = 6`

也就是：

- 每个输出帧会拼接连续 `7` 帧 Mel 特征
- 每次向前滑动 `6` 帧
- 左侧会做 `(7 - 1) / 2 = 3` 帧补齐

因为基础 fbank 是 `80` 维，所以 LFR 后单帧维度变成：

- `80 × 7 = 560`

这也是代码里固定 `featureDim = 560` 的来源。

### 5. 固定长度对齐

`SenseVoiceSmall300` 这版 Core ML 模型当前按固定时间维输入处理，代码里写死为：

- `fixedFrames = 300`

因此 LFR 输出后会做一次长度对齐：

- 如果帧数大于 `300`，直接截断到前 `300` 帧
- 如果帧数小于 `300`，在尾部补零帧到 `300` 帧

最终送入模型的张量形状为：

- `shape = [1, 300, 560]`
- 数据类型为 `float32`

### 6. 模型输出与解码

当前推理只依赖两个输出：

- `ctc_logits`
- `encoder_out_lens`

解码方式是最简单直接的 CTC greedy decode：

- 每个时间步取概率最大的 token id
- 删除连续重复 token
- 删除 `blankID = 0`
- 跳过 `<blank>`, `<blk>`, `<s>`, `</s>`, `<unk>`, `<pad>` 等特殊 token
- 最后直接把 token 字符串拼接成原始文本

### 7. 文本后处理

greedy decode 之后，还会做一轮 SenseVoice 标签清理：

- 移除语言标签，如 `<|zh|>`, `<|en|>`, `<|ja|>` 等
- 移除情绪和事件标签，如 `<|NEUTRAL|>`, `<|BGM|>`, `<|Speech|>` 等
- 将 sentencepiece 风格的 `▁` 替换为空格
- 用正则压缩多余空白
- 最后去掉首尾空格和换行

所以页面上会同时保留：

- `Raw Text`：模型解码后的原始文本
- `Final Text`：做完标签清理和空白整理后的最终文本

## 目前状态

- 已完成基础的本地录音与离线识别流程。
- 已将 UI、状态管理、音频处理、前后处理和错误定义分离。
- 当前仍以单页 Demo 形式存在，适合继续往产品化方向演进。

## ToDo List

- 为转写、前处理、后处理补充单元测试。
- 清理仓库中的历史实验文件和重复录音实现，例如 `MicrophoneRecorder.swift.swift`。
- 把模型配置、固定帧长、采样率等参数做成集中配置项。
- 补充当前 `convert_coreml` 相对上游版本的具体改动说明。
- 给界面增加更清晰的运行状态和错误提示。
- 支持选择本地音频文件，而不只依赖 `test.wav` 和麦克风录音。
- 评估将推理任务放到后台队列，减少主线程阻塞风险。
- 为识别结果增加导出、复制和历史记录能力。
