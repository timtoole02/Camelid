// Capture only. All resampling and speech inference happen in the Rust backend.
class CamelidRecorder extends AudioWorkletProcessor {
  constructor() {
    super()
    this.active = true
    this.buffer = new Float32Array(4096)
    this.offset = 0
    this.port.onmessage = ({ data }) => {
      if (data === 'stop') {
        this.active = false
        this.flush()
        this.port.postMessage({ stopped: true })
      }
    }
  }
  flush() {
    if (!this.offset) return
    const samples = this.buffer.slice(0, this.offset)
    this.port.postMessage({ samples }, [samples.buffer])
    this.offset = 0
  }
  process(inputs) {
    if (!this.active) return true
    const channels = inputs[0]
    if (!channels?.length) return true
    for (let i = 0; i < channels[0].length; i += 1) {
      let sample = 0
      for (const channel of channels) sample += channel[i] || 0
      this.buffer[this.offset++] = sample / channels.length
      if (this.offset === this.buffer.length) this.flush()
    }
    return true
  }
}
registerProcessor('camelid-recorder', CamelidRecorder)
