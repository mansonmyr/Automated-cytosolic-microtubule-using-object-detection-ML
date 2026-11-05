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
// Letterbox the whole source. Round the interpolated RGB byte before normalization,
// matching the native uint8 resize for 2048 x 2048 TIFFs at input size 1024.
function wholeImageInput(rgba, width, height, size) {
  const scale = Math.min(size/width, size/height);
  const w = Math.max(1, Math.round(width*scale)), h = Math.max(1, Math.round(height*scale));
  const left = Math.floor((size-w)/2), top = Math.floor((size-h)/2), plane = size*size;
  const input = new Float32Array(3*plane);
  if (w !== size || h !== size) input.fill(114/255);
  for (let y=0; y<h; y++) {
    const sy=Math.max(0,Math.min(height-1,(y+.5)*height/h-.5)), y0=Math.floor(sy), y1=Math.min(y0+1,height-1);
    const fy=sy-y0;
    for (let x=0; x<w; x++) {
      const sx=Math.max(0,Math.min(width-1,(x+.5)*width/w-.5)), x0=Math.floor(sx), x1=Math.min(x0+1,width-1);
      const fx=sx-x0, dst=(y+top)*size+x+left;
      for (let c=0; c<3; c++) {
        if (width===2*w && height===2*h) {
          // This OpenCV uint8 half-size path rounds vertical pairs, then horizontal pairs.
          const v0=(rgba[4*(y0*width+x0)+c]+rgba[4*(y1*width+x0)+c]+1)>>1;
          const v1=(rgba[4*(y0*width+x1)+c]+rgba[4*(y1*width+x1)+c]+1)>>1;
          input[c*plane+dst]=((v0+v1+1)>>1)/255;
          continue;
        }
        const a=rgba[4*(y0*width+x0)+c]*(1-fx)+rgba[4*(y0*width+x1)+c]*fx;
        const b=rgba[4*(y1*width+x0)+c]*(1-fx)+rgba[4*(y1*width+x1)+c]*fx;
        input[c*plane+dst]=Math.round(a*(1-fy)+b*fy)/255;
      }
    }
  }
  return {input, mapBox: box => box.map((v,i) => i%2
    ? Math.min(height,Math.max(0,(v-top)*height/h))
    : Math.min(width,Math.max(0,(v-left)*width/w)))};
}
// Each flipped detection can support at most one original detection.
function mergeFlipAgreement(original, flipped) {
  const used = new Set();
  const merged = original.map((detection) => {
    let match = -1;
    let bestOverlap = 0.5;
    flipped.forEach((candidate, index) => {
      if (used.has(index)) return;
      const overlap = intersectionOverUnion(detection.box, candidate.box);
      if (overlap >= 0.5 && (match < 0 || overlap > bestOverlap)) {
        match = index;
        bestOverlap = overlap;
      }
    });
    if (match < 0) return {...detection, score: detection.score / 2};
    used.add(match);
    const partner = flipped[match];
    return {
      ...detection,
      box: detection.box.map((coordinate, index) => (coordinate + partner.box[index]) / 2),
      score: (detection.score + partner.score) / 2,
    };
  });
  for (const [index, detection] of flipped.entries()) {
    if (!used.has(index)) merged.push({...detection, score: detection.score / 2});
  }
  return suppressOverlaps(merged);
}
if (typeof module !== 'undefined') module.exports = {tilePositions, intersectionOverUnion, suppressOverlaps, wholeImageInput, mergeFlipAgreement};
