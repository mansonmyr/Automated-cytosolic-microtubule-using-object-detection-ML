// Coordinates are in native image pixels; only the outer edges are padded.
function tilePositions(length, size = 640, overlap = 160) {
  if (![length, size, overlap].every(Number.isFinite) || length <= 0 || size <= 0 || overlap < 0 || overlap >= size) {
    throw new Error('Invalid tile dimensions or overlap.');
  }
  if (length <= size) return [0];
  const points = [];
  for (let p = 0; p < length - size; p += size - overlap) points.push(p);
  points.push(length - size);
  return points;
}
function intersectionOverUnion(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
  return union > 0 ? inter / union : 0;
}
function suppressOverlaps(items, threshold = 0.5) {
  const kept = [];
  for (const candidate of [...items].sort ( (a,b) => b.score-a.score)) {
    if (!kept.some(d => d.id === candidate.id && intersectionOverUnion(d.box, candidate.box) > threshold)) kept.push(candidate);
  }
  return kept;
}
if (typeof module !== 'undefined') module.exports = {tilePositions, intersectionOverUnion, suppressOverlaps};
