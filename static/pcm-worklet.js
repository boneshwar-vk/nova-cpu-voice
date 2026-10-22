// AudioWorklet: mono float input -> 16 kHz Int16 frames of 512 samples (32 ms),
// the exact frame the server's VAD consumes. Resamples if the browser refused a
// 16 kHz AudioContext (Safari). Posts {pcm: ArrayBuffer, peak: number} per frame.
class Pcm16Frames extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000; // 1 when the context runs at 16 kHz
    this.acc = [];                    // pending 16 kHz float samples
    this.phase = 0;                   // fractional read position for resampling
    this.prev = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    if (this.ratio === 1) {
      for (let i = 0; i < ch.length; i++) this.acc.push(ch[i]);
    } else {
      // linear interpolation down/up-sampling to 16 kHz
      let pos = this.phase;
      while (pos < ch.length) {
        const i = Math.floor(pos);
        const frac = pos - i;
        const a = i === 0 ? this.prev : ch[i - 1];
        const b = ch[Math.min(i, ch.length - 1)];
        this.acc.push(a + (b - a) * frac);
        pos += this.ratio;
      }
      this.phase = pos - ch.length;
      this.prev = ch[ch.length - 1];
    }
    while (this.acc.length >= 512) {
      const frame = this.acc.splice(0, 512);
      const out = new Int16Array(512);
      let peak = 0;
      for (let i = 0; i < 512; i++) {
        const s = Math.max(-1, Math.min(1, frame[i]));
        out[i] = s < 0 ? s * 32768 : s * 32767;
        const a = Math.abs(s);
        if (a > peak) peak = a;
      }
      this.port.postMessage({ pcm: out.buffer, peak }, [out.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm16-frames', Pcm16Frames);
