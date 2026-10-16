// Dependency-free deterministic RGB fixture: blue left half, yellow right half.
import { deflateSync } from 'node:zlib'
import { writeFile } from 'node:fs/promises'
function crc32(bytes) {
  let crc = 0xffffffff
  for (const byte of bytes) {
    crc ^= byte
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0)
  }
  return (crc ^ 0xffffffff) >>> 0
}
function chunk(type, data) {
  const name = Buffer.from(type), size = Buffer.alloc(4), crc = Buffer.alloc(4)
  size.writeUInt32BE(data.length)
  crc.writeUInt32BE(crc32(Buffer.concat([name, data])))
  return Buffer.concat([size, name, data, crc])
}
const width = 96, height = 64
const pixels = Buffer.alloc(height * (1 + width * 3))
for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
  const offset = y * (1 + width * 3) + 1 + x * 3
  pixels.set(x < width / 2 ? [0, 0, 255] : [255, 255, 0], offset)
}
const header = Buffer.alloc(13)
header.writeUInt32BE(width, 0); header.writeUInt32BE(height, 4)
header[8] = 8; header[9] = 2
await writeFile(new URL('./blue-yellow.png', import.meta.url), Buffer.concat([
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
  chunk('IHDR', header), chunk('IDAT', deflateSync(pixels)), chunk('IEND', Buffer.alloc(0)),
]))
