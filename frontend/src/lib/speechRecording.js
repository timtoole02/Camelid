import workletUrl from './speech-worklet.js?url&no-inline'

export const MAX_RECORDING_SECONDS = 30
export function encodeWav(chunks, sampleRate, count) {
  const buffer = new ArrayBuffer(44 + count * 2)
  const view = new DataView(buffer)
  const ascii = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)))
  ascii(0, 'RIFF'); view.setUint32(4, 36 + count * 2, true); ascii(8, 'WAVE')
  ascii(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true)
  ascii(36, 'data'); view.setUint32(40, count * 2, true)
  let offset = 44
  for (const chunk of chunks) {
    for (const value of chunk) {
      if (offset >= buffer.byteLength) break
      const sample = Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0
      view.setInt16(offset, Math.round(sample * (sample < 0 ? 32768 : 32767)), true)
      offset += 2
    }
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

export function microphoneSupport() {
  if (!window.isSecureContext) return 'Microphone access needs HTTPS or localhost. Open Camelid on this computer or use an HTTPS connection.'
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext || !window.AudioWorkletNode) return 'This browser does not support microphone recording. Use a current browser or Camelid Desktop.'
  return ''
}

export class SpeechRecording {
  constructor(onLimit, onSeconds, onError) {
    this.onLimit = onLimit
    this.onSeconds = onSeconds
    this.onError = onError
    this.chunks = []
    this.count = 0
    this.closed = false
    this.stopping = false
  }
  async start() {
    const unsupported = microphoneSupport()
    if (unsupported) throw new Error(unsupported)
    this.context = new AudioContext()
    await this.context.resume()
    if (this.closed) return
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }, video: false })
    if (this.closed) { stream.getTracks().forEach((track) => track.stop()); return }
    this.stream = stream
    for (const track of stream.getAudioTracks()) {
      track.onended = () => { if (!this.closed && !this.stopping) this.onError(new Error('The microphone disconnected. Please try again.')) }
    }
    await this.context.audioWorklet.addModule(workletUrl)
    if (this.closed) return
    this.sampleRate = this.context.sampleRate
    this.node = new AudioWorkletNode(this.context, 'camelid-recorder')
    this.node.onprocessorerror = () => this.onError(new Error('Microphone recording failed. Please try again.'))
    this.node.port.onmessage = ({ data }) => {
      if (this.closed) return
      if (data.stopped) { this.resolveStop?.(); return }
      if (!data.samples) return
      const remaining = this.sampleRate * MAX_RECORDING_SECONDS - this.count
      const chunk = data.samples.subarray(0, remaining)
      this.chunks.push(chunk)
      this.count += chunk.length
      this.onSeconds(Math.floor(this.count / this.sampleRate))
      if (this.count >= this.sampleRate * MAX_RECORDING_SECONDS && !this.stopping) {
        this.stopping = true
        this.onLimit()
      }
    }
    this.source = this.context.createMediaStreamSource(stream)
    this.gain = this.context.createGain()
    this.gain.gain.value = 0
    this.source.connect(this.node).connect(this.gain).connect(this.context.destination)
    // Bound capture even if a suspended audio context stops delivering samples.
    this.timer = setTimeout(() => { if (!this.closed && !this.stopping) { this.stopping = true; this.onLimit() } }, MAX_RECORDING_SECONDS * 1000)
  }
  async stop() {
    this.stopping = true
    clearTimeout(this.timer)
    this.stream?.getTracks().forEach((track) => track.stop())
    try {
      if (!this.node || this.closed) throw new Error('No recording is available.')
      await new Promise((resolve, reject) => {
        this.stopTimer = setTimeout(() => reject(new Error('Could not finish recording. Please try again.')), 1500)
        this.resolveStop = resolve
        this.node.port.postMessage('stop')
      })
      if (this.count < this.sampleRate / 5) throw new Error('Recording is too short. Speak for at least a moment.')
      return encodeWav(this.chunks, this.sampleRate, this.count)
    } finally { this.close() }
  }
  close() {
    this.closed = true
    clearTimeout(this.timer)
    clearTimeout(this.stopTimer)
    this.resolveStop?.()
    this.stream?.getTracks().forEach((track) => track.stop())
    this.source?.disconnect()
    this.node?.disconnect()
    this.node?.port.close()
    this.gain?.disconnect()
    this.context?.close().catch(() => {})
    this.chunks = []
  }
}
