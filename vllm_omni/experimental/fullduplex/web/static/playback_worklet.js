class FullDuplexPcmPlayback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.playedFrames = 0;
    this.drain = null;
    this.started = false;
    this.port.onmessage = (event) => this.handleMessage(event.data || {});
  }

  handleMessage(message) {
    if (message.type === 'audio' && message.pcm) {
      this.queue.push(message.pcm);
      if (!this.started) {
        this.started = true;
        this.port.postMessage({ type: 'playback-started' });
      }
    } else if (message.type === 'drain') {
      this.drain = { responseId: message.responseId || null };
      this.notifyIfDrained();
    } else if (message.type === 'clear') {
      this.queue = [];
      this.offset = 0;
      this.playedFrames = 0;
      this.drain = null;
      this.started = false;
    }
  }

  notifyIfDrained() {
    if (!this.drain || this.queue.length > 0) return;
    this.port.postMessage({
      type: 'playback-drained',
      responseId: this.drain.responseId,
      playedMs: Math.round((this.playedFrames * 1000) / sampleRate),
    });
    this.playedFrames = 0;
    this.drain = null;
    this.started = false;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
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
    this.notifyIfDrained();
    return true;
  }
}

registerProcessor('fullduplex-pcm-playback', FullDuplexPcmPlayback);
