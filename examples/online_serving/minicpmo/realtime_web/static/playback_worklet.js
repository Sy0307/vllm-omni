class FullDuplexPcmPlayback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.playedFrames = 0;
    this.underrunFrames = 0;
    this.drain = null;
    this.started = false;
    this.activeResponseId = null;
    this.initialBufferFrames = Math.round(sampleRate * 0.2);
    this.port.onmessage = (event) => this.handleMessage(event.data || {});
  }

  handleMessage(message) {
    if (message.type === 'audio' && message.pcm) {
      if (!this.started && !this.activeResponseId) {
        this.activeResponseId = message.responseId || null;
      }
      if (!this.started && Number.isFinite(message.initialBufferMs)) {
        this.initialBufferFrames = Math.max(0, Math.round((sampleRate * message.initialBufferMs) / 1000));
      }
      this.queue.push(message.pcm);
      this.startIfBuffered();
    } else if (message.type === 'drain') {
      this.drain = { responseId: message.responseId || null };
      if (!this.started && this.bufferedFrames() > 0) this.startPlayback();
      this.notifyIfDrained();
    } else if (message.type === 'clear') {
      this.queue = [];
      this.offset = 0;
      this.playedFrames = 0;
      this.underrunFrames = 0;
      this.drain = null;
      this.started = false;
      this.activeResponseId = null;
    }
  }

  bufferedFrames() {
    return this.queue.reduce((total, pcm, index) => (
      total + pcm.length - (index === 0 ? this.offset : 0)
    ), 0);
  }

  startPlayback() {
    if (this.started) return;
    this.started = true;
    this.port.postMessage({ type: 'playback-started', responseId: this.activeResponseId });
  }

  startIfBuffered() {
    if (!this.started && this.bufferedFrames() >= this.initialBufferFrames) this.startPlayback();
  }

  notifyIfDrained() {
    if (!this.drain || this.queue.length > 0) return;
    this.port.postMessage({
      type: 'playback-drained',
      responseId: this.drain.responseId,
      playedMs: Math.round((this.playedFrames * 1000) / sampleRate),
      underrunMs: Math.round((this.underrunFrames * 1000) / sampleRate),
    });
    this.playedFrames = 0;
    this.underrunFrames = 0;
    this.drain = null;
    this.started = false;
    this.activeResponseId = null;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    if (!this.started) {
      this.notifyIfDrained();
      return true;
    }
    let target = 0;
    while (target < output.length && this.queue.length > 0) {
      const pcm = this.queue[0];
      const count = Math.min(output.length - target, pcm.length - this.offset);
      for (let index = 0; index < count; index += 1) {
        output[target + index] = pcm[this.offset + index] / 32768;
      }
      target += count;
      this.offset += count;
      this.playedFrames += count;
      if (this.offset >= pcm.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (target < output.length && !this.drain) {
      this.underrunFrames += output.length - target;
      this.port.postMessage({
        type: 'playback-underrun',
        responseId: this.activeResponseId,
        underrunMs: Math.round((this.underrunFrames * 1000) / sampleRate),
      });
    }
    this.notifyIfDrained();
    return true;
  }
}

registerProcessor('fullduplex-pcm-playback', FullDuplexPcmPlayback);
