export function inspectGlb(bytes) {
  const view = new DataView(bytes);
  const word = (offset) => view.getUint32(offset, true);
  if (bytes.byteLength < 20 || word(0) !== 0x46546c67 || word(4) !== 2
      || word(8) !== bytes.byteLength) {
    throw new Error("Invalid GLB 2.0 header.");
  }
  let json;
  const chunks = [];
  for (let offset = 12; offset < bytes.byteLength;) {
    if (offset + 8 > bytes.byteLength) throw new Error("Truncated GLB chunk header.");
    const length = word(offset);
    const type = word(offset + 4);
    const start = offset + 8;
    if (length % 4 || start + length > bytes.byteLength) throw new Error("Invalid GLB chunk length.");
    if (offset === 12 && type !== 0x4e4f534a) throw new Error("Missing GLB JSON chunk.");
    const name = type === 0x4e4f534a ? "JSON" : type === 0x004e4942 ? "BIN" : "Other";
    if (offset === 12) json = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, start, length)));
    chunks.push(`${name}: ${length} bytes`);
    offset = start + length;
  }
  if (!json || typeof json !== "object" || json.asset?.version !== "2.0") {
    throw new Error("Missing glTF 2.0 asset description.");
  }
  return { json, chunks };
}
