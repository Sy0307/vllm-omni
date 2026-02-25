# RFC #1479：WebSocket TTS 流式音频输出（Streaming PCM）实现计划

目标：在现有 **WebSocket 流式文本输入**（PR #1230）基础上，增加 **流式音频输出**：同一句话的音频不再等整句合成完才一次性返回，而是按 Code2Wav 解码窗口持续推送 **多帧 PCM chunk**（binary WebSocket frames），使客户端可以边收边播。

本分支基线：`dev/streaming_ws_tts` 已合并 `pr-1230`（WS 流式文本输入）+ `pr-1438`（HTTP `stream=true` 的 PCM 流式输出），用于在同一代码基线下实现 #1479。

---

## 现状与差距

### 已有能力
- **WS 文本流式输入**：`/v1/audio/speech/stream` 支持 `input.text` 逐段输入、按句切分、逐句生成（PR #1230）。
- **HTTP 音频流式输出**：`POST /v1/audio/speech` 支持 `stream=true` + `response_format="pcm"`，边解码边输出 PCM chunk（PR #1438）。

### 需要补齐的能力（#1479）
- **WS 音频流式输出**：在 `audio.start` 与 `audio.done` 之间，发送 *多个* binary PCM chunk frame（而不是 1 个完整音频 frame）。

---

## 协议设计（对齐 RFC #1479）

### Client → Server
`session.config` 新增字段：
- `stream_audio: bool`（默认 `false`，保持完全向后兼容）

示例：
```json
{"type":"session.config","voice":"Vivian","stream_audio":true,"response_format":"pcm"}
```

### Server → Client（按句）
当 `stream_audio=false`（默认）：保持旧行为
```
audio.start
<binary: 整句音频>
audio.done
```

当 `stream_audio=true`：改为多 chunk
```
audio.start (format=pcm, sample_rate=24000)
<binary: chunk 1>
<binary: chunk 2>
...
audio.done (total_bytes=...)
```

约束（建议在 WS handler 中显式校验并失败早返回）：
- `stream_audio=true` ⇒ `response_format` 必须是 `"pcm"`
- `stream_audio=true` ⇒ `speed` 必须是 `1.0`（或省略），避免引入“流式边播边变速”的复杂度

---

## 实现方案（推荐）

### 1) 配置层：WS session.config 增加 `stream_audio`
- 修改 `StreamingSpeechSessionConfig` 增加 `stream_audio: bool = False`
- 默认关闭以保证所有现有客户端不变

### 2) 复用 pr-1438 的流式生成路径（关键）
WS handler 不能直接复用 `create_speech(stream=True)`（它返回 `StreamingResponse`），建议做“可复用的内部生成器”：

推荐重构方向（实现时二选一，倾向 A）：
- A. 在 `OmniOpenAIServingSpeech` 增加一个内部方法：给定 `OpenAICreateSpeechRequest`，返回 `AsyncIterator[bytes]`（PCM chunk bytes）
  - HTTP：`create_speech()` 调用该迭代器并包一层 `StreamingResponse`
  - WS：handler 直接 `async for chunk in iterator: websocket.send_bytes(chunk)`
- B. 直接在 WS handler 内复刻 pr-1438 的 `_generate_pcm_chunks(generator, request_id)` + `engine_client.generate(...)` 逻辑
  - 代码更快落地，但会产生逻辑重复（后续维护成本更高）

必须处理的细节：
- **delta slicing**：engine 输出可能是「累积 list」或「每步单 tensor」两种模式；必须只发送新增 chunk，避免重复发送（pr-1438 已覆盖该逻辑）。
- **sample_rate**：尽量从输出里读 `sr`；拿不到时默认 24000（与模型约定一致）。

### 3) WS handler：在 `audio.start`/`audio.done` 之间发送多个 binary frame
改造点：`OmniStreamingSpeechHandler._generate_and_send(...)`
- `stream_audio=false`：保持当前“一句一次性 bytes”路径
- `stream_audio=true`：
  - 先发 `audio.start`（带 `format=pcm`、`sample_rate`）
  - 迭代产出 PCM chunk：每个 chunk 调用 `websocket.send_bytes(chunk)`
  - 累积 `total_bytes`
  - 结束时发 `audio.done`（带 `total_bytes`）

### 4) 示例客户端与文档
- 更新 `streaming_speech_client.py`：当 `stream_audio=true` 时对同一句的多 chunk 做 append（当前示例“收到一次 binary 就写一个文件”会覆盖/丢前面 chunk）。
- 文档补充 WS `stream_audio` 行为与约束（仅 PCM、speed=1.0）。

---

## 测试策略（最小但覆盖关键行为）

建议扩展现有 WS 测试（PR #1230 已有）：
- `stream_audio=false`：仍然每句只收到 1 个 binary frame（回归）
- `stream_audio=true`：
  - `audio.start` 后可连续 `receive_bytes()` 多次
  - `audio.done.total_bytes == sum(len(chunk_i))`
  - 非 pcm / speed!=1.0 时返回 `error` 并且不发送 binary

---

## 预估改动量（实现 #1479 本身，不含本分支已合并依赖）

粗略估算（取“中位数”）：
- `vllm_omni/entrypoints/openai/protocol/audio.py`：+10 ~ +25 行（`stream_audio` 配置字段 + 校验）
- `vllm_omni/entrypoints/openai/serving_speech_stream.py`：+80 ~ +180 行（WS handler chunked send、计数、错误处理）
- `vllm_omni/entrypoints/openai/serving_speech.py`：+30 ~ +120 行（提取可复用 streaming generator，或轻量复用 hook）
- `tests/entrypoints/openai_api/test_serving_speech_stream.py`：+60 ~ +140 行（WS streaming 音频输出测试）
- `examples/online_serving/qwen3_tts/streaming_speech_client.py`：+30 ~ +80 行（append/拼接逻辑）
- 文档：+20 ~ +60 行

合计：大约 **+230 ~ +600 行**（视“是否做 serving_speech 重构复用”而变化）。

---

## 需要提前决策的问题（建议默认方案）

1) **`audio.start` 的发送时机**
- 选项 A：收到句子后立刻发（sample_rate 写死 24000 或缺省）
- 选项 B：等到第一个 chunk 到来再发（能携带真实 sr，但首包 JSON 会晚一点）
- 建议：A（更简单，且 sr 通常固定 24000）

2) **错误语义**
- 选项 A：句子内出错也发 `audio.done`（客户端按句推进不阻塞）
- 选项 B：出错直接关闭 session
- 建议：A（与 PR #1230 行为一致：non-fatal error）

3) **chunk 粒度控制**
- 是否允许 WS 侧再做二次切分（按字节/毫秒）？
- 建议：不做；**严格跟随模型 codec window**（通过 stage config `codec_chunk_frames` / `codec_left_context_frames` 控制）

4) **是否支持 WAV 流式**
- 建议：不支持（WAV header 依赖总长度；若要支持需额外 framing 或“伪 WAV + 修 header”，复杂且收益低）

5) **与 HTTP `stream=true` 的一致性**
- WS `stream_audio=true` 是否复用同一套 Pydantic 约束（pcm + speed=1.0）？
- 建议：一致，减少用户困惑

---

## 实施里程碑（建议拆分 PR）

1. PR1（小）：WS `session.config` 增加 `stream_audio` + 文档/示例声明（但仍走旧的整句返回）
2. PR2（核心）：WS handler 接入 streaming generator，binary chunk 帧输出 + 测试
3. PR3（可选）：serving_speech 代码去重/重构、更多边界测试（断连 cancel / backpressure）

