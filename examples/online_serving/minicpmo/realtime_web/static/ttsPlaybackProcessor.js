// tts-playback-processor — jitter-buffered FIFO player for Int16Array chunks.
// The UI resamples incoming model audio to this AudioContext's sample rate.
// postMessage({type:'clear'}) flushes the queue for barge-in.
class TTSPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferQueue = [];
    this.readOffset = 0;
    this.samplesRemaining = 0;
    this.isPlaying = false;
    this.drainRequested = false;
    this.drainGraceSamples = Math.round(sampleRate * 0.7);
    this.drainWaitSamples = 0;
    this.gapFillSamples = Math.round(sampleRate * 0.5);
    this.gapRemainingSamples = 0;
    this.underrunCount = 0;
    this.playedSamplesTotal = 0;
    this.baseMinStartSamples = Math.round(sampleRate * 1.0);
    this.minStartSamples = this.baseMinStartSamples;
    this.maxMinStartSamples = Math.round(sampleRate * 2.0);
    this.port.onmessage = (e) => {
      if (e.data && typeof e.data === 'object' && e.data.type === 'clear') {
        this.bufferQueue = [];
        this.readOffset = 0;
        this.samplesRemaining = 0;
        this.isPlaying = false;
        this.drainRequested = false;
        this.drainWaitSamples = 0;
        this.gapRemainingSamples = 0;
        this.minStartSamples = this.baseMinStartSamples;
        return;
      }
      if (e.data && typeof e.data === 'object' && e.data.type === 'config') {
        const ms = Number(e.data.minStartMs);
        if (Number.isFinite(ms) && ms >= 0) {
          this.baseMinStartSamples = Math.round(sampleRate * ms / 1000);
          this.minStartSamples = this.baseMinStartSamples;
          this.maxMinStartSamples = Math.max(this.minStartSamples, Math.round(sampleRate * 2.0));
        }
        const maxMs = Number(e.data.maxStartMs);
        if (Number.isFinite(maxMs) && maxMs >= 0) {
          this.maxMinStartSamples = Math.max(this.minStartSamples, Math.round(sampleRate * maxMs / 1000));
        }
        const drainMs = Number(e.data.drainGraceMs);
        if (Number.isFinite(drainMs) && drainMs >= 0) this.drainGraceSamples = Math.round(sampleRate * drainMs / 1000);
        const gapMs = Number(e.data.gapFillMs);
        if (Number.isFinite(gapMs) && gapMs >= 0) this.gapFillSamples = Math.round(sampleRate * gapMs / 1000);
        return;
      }
      if (e.data && typeof e.data === 'object' && e.data.type === 'drain') {
        this.drainRequested = true;
        this.drainWaitSamples = this.drainGraceSamples;
        return;
      }
      const chunk = e.data && typeof e.data === 'object' && e.data.type === 'audio' ? e.data.pcm : e.data;
      if (!chunk || !chunk.length) return;
      if (this.samplesRemaining === 0 && this.isPlaying) this.gapRemainingSamples = 0;
      this.bufferQueue.push(chunk);
      this.samplesRemaining += chunk.length;
    };
  }
  process(inputs, outputs) {
    const out = outputs[0][0];
    if (this.samplesRemaining === 0) {
      out.fill(0);
      if (this.isPlaying) {
        if (this.drainRequested) {
          this.drainWaitSamples = Math.max(0, this.drainWaitSamples - out.length);
          if (this.drainWaitSamples > 0) return true;
          this.isPlaying = false;
          this.drainRequested = false;
          this.gapRemainingSamples = 0;
          this.port.postMessage({
            type: 'ttsPlaybackStopped',
            playedMs: Math.round(this.playedSamplesTotal * 1000 / sampleRate),
          });
          return true;
        }
        if (this.gapRemainingSamples > 0) {
          this.gapRemainingSamples = Math.max(0, this.gapRemainingSamples - out.length);
          return true;
        }
        this.underrunCount++;
        this.isPlaying = false;
        this.minStartSamples = Math.min(this.maxMinStartSamples, Math.max(this.minStartSamples * 1.25, this.baseMinStartSamples));
        this.port.postMessage({
          type: 'ttsPlaybackUnderrun',
          count: this.underrunCount,
          minStartMs: Math.round(this.minStartSamples * 1000 / sampleRate),
        });
        this.port.postMessage({
          type: 'ttsPlaybackStopped',
          playedMs: Math.round(this.playedSamplesTotal * 1000 / sampleRate),
        });
      }
      return true;
    }
    if (!this.isPlaying && !this.drainRequested && this.samplesRemaining < this.minStartSamples) {
      out.fill(0);
      return true;
    }
    if (!this.isPlaying && this.samplesRemaining === 0) {
      out.fill(0);
      return true;
    }
    if (!this.isPlaying) {
      this.isPlaying = true;
      this.port.postMessage({ type: 'ttsPlaybackStarted' });
    }
    let i = 0;
    while (i < out.length && this.bufferQueue.length > 0) {
      const b = this.bufferQueue[0];
      out[i++] = b[this.readOffset] / 32768;
      this.readOffset++;
      this.samplesRemaining--;
      this.playedSamplesTotal++;
      if (this.readOffset >= b.length) {
        this.bufferQueue.shift();
        this.readOffset = 0;
      }
    }
    if (i < out.length) {
      while (i < out.length) out[i++] = 0;
      if (this.drainRequested) {
        this.isPlaying = false;
        this.drainRequested = false;
        this.drainWaitSamples = 0;
        this.gapRemainingSamples = 0;
        this.port.postMessage({
          type: 'ttsPlaybackStopped',
          playedMs: Math.round(this.playedSamplesTotal * 1000 / sampleRate),
        });
        return true;
      }
      this.underrunCount++;
      this.isPlaying = false;
      this.drainRequested = false;
      this.drainWaitSamples = 0;
      this.gapRemainingSamples = 0;
      this.minStartSamples = Math.min(this.maxMinStartSamples, Math.max(this.minStartSamples * 1.5, sampleRate));
      this.port.postMessage({
        type: 'ttsPlaybackUnderrun',
        count: this.underrunCount,
        minStartMs: Math.round(this.minStartSamples * 1000 / sampleRate),
      });
      this.port.postMessage({
        type: 'ttsPlaybackStopped',
        playedMs: Math.round(this.playedSamplesTotal * 1000 / sampleRate),
      });
      return true;
    }
    if (this.samplesRemaining === 0) {
      this.gapRemainingSamples = this.drainRequested ? 0 : this.gapFillSamples;
    }
    return true;
  }
}
registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
