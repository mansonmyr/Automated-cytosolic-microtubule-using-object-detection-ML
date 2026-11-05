/* Geometry and annotation state always use original image pixels. */
(function (root) {
  'use strict';
  const names = ['cytosolic', 'centrosomal', 'cell'];
  const clone = items => items.map(d => ({...d, box: [...d.box]}));
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  function imagePoint(clientX, clientY, rect, width, height) {
    return [clamp((clientX - rect.left) * width / rect.width, 0, width),
      clamp((clientY - rect.top) * height / rect.height, 0, height)];
  }

  function normalizedBox(a, b, width, height) {
    if (![...a, ...b, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;
    const box = [clamp(Math.min(a[0], b[0]), 0, width), clamp(Math.min(a[1], b[1]), 0, height),
      clamp(Math.max(a[0], b[0]), 0, width), clamp(Math.max(a[1], b[1]), 0, height)];
    return box[2] - box[0] >= 2 && box[3] - box[1] >= 2 ? box : null;
  }

  function hitTest(point, items) {
    // Small boxes remain selectable inside the larger cell boxes.
    const hits = items.filter(d => point[0] >= d.box[0] && point[0] <= d.box[2]
      && point[1] >= d.box[1] && point[1] <= d.box[3]);
    hits.sort((a, b) => (a.box[2]-a.box[0])*(a.box[3]-a.box[1])
      - (b.box[2]-b.box[0])*(b.box[3]-b.box[1]));
    return hits[0]?.uid ?? null;
  }

  function validateReview(review, width, height, fileHash) {
    if (!review || review.schema_version !== 2) throw Error('Only version 2 review files support resuming. Older JSON files remain audit exports.');
    if (review.image_width !== width || review.image_height !== height) throw Error('The review has different image dimensions.');
    if (typeof fileHash !== 'string' || !/^[a-f0-9]{64}$/.test(fileHash)
      || review.source_file_sha256 !== fileHash) throw Error('This review belongs to a different image file. Open its original image first.');
    if (!Number.isFinite(review.confidence_threshold) || review.confidence_threshold < 0.05 || review.confidence_threshold > 0.95
      || Math.abs(review.confidence_threshold*1000-Math.round(review.confidence_threshold*1000)) > 1e-6) {
      throw Error('The saved confidence threshold is invalid.');
    }
    function cleanItems(items, originals = false) {
      if (!Array.isArray(items) || items.length > 50000) throw Error('The saved boxes are invalid or exceed the supported limit.');
      const ids = new Set();
      return items.map(d => {
        if (!d || !['model','manual'].includes(d.origin) || !Number.isInteger(d.id) || d.id < 0 || d.id > 2
          || typeof d.uid !== 'string' || !/^(model|manual)-[1-9]\d{0,8}$/.test(d.uid)
          || !d.uid.startsWith(d.origin+'-') || ids.has(d.uid)
          || (originals && d.origin !== 'model') || !Array.isArray(d.box) || d.box.length !== 4
          || !d.box.every(Number.isFinite) || d.box[0] < 0 || d.box[1] < 0 || d.box[2] > width || d.box[3] > height
          || d.box[2] <= d.box[0] || d.box[3] <= d.box[1]
          || (d.origin === 'manual' ? d.score !== null : !Number.isFinite(d.score) || d.score < 0.05 || d.score > 1)) {
          throw Error('A saved box has invalid coordinates, class or provenance.');
        }
        ids.add(d.uid);
        return {id:d.id, uid:d.uid, box:[...d.box], score:d.score, origin:d.origin};
      });
    }
    const original = cleanItems(review.original_predictions, true), items = cleanItems(review.all_detections);
    const originalById = new Map(original.map(d => [d.uid,d]));
    for (const item of items.filter(d => d.origin === 'model')) {
      const previous = originalById.get(item.uid);
      if (!previous || previous.id !== item.id || previous.score !== item.score
        || !previous.box.every((v,i) => v === item.box[i])) throw Error('A retained prediction disagrees with the original prediction record.');
    }
    let model = null;
    if (review.model !== null) {
      const m = review.model;
      if (!m || typeof m.name !== 'string' || m.name.length > 200 || typeof m.url !== 'string' || m.url.length > 2000
        || !Number.isInteger(m.inputSize) || m.inputSize < 32 || m.inputSize > 4096 || m.inputSize%32
        || !Number.isInteger(m.overlap) || m.overlap < 0 || m.overlap >= m.inputSize
        || JSON.stringify(m.classes) !== JSON.stringify(names)) throw Error('The saved model description is invalid.');
      // Provenance only. Import never downloads or runs the model named in a review.
      model = {name:m.name,url:m.url,inputSize:m.inputSize,overlap:m.overlap,classes:[...names],...(m.mode === 'whole_image' ? {mode:m.mode} : {})};
    }
    if (original.length && !model) throw Error('The original predictions have no model description.');
    return {items,original,model,cutoff:review.confidence_threshold};
  }

  class ReviewStore {
    constructor() { this.reset(); }
    reset(predictions = []) {
      this.items = predictions.map((d, i) => ({...d, box: [...d.box], uid: `model-${i+1}`, origin: 'model'}));
      this.original = clone(this.items);
      this.history = [];
      this.selected = null;
      this.nextManual = 1;
    }
    visible(cutoff) { return this.items.filter(d => d.origin === 'manual' || d.score >= cutoff); }
    remember() {
      this.history.push({items: clone(this.items), selected: this.selected});
      if (this.history.length > 50) this.history.shift();
    }
    add(id, a, b, width, height) {
      const box = normalizedBox(a, b, width, height);
      if (!box || !Number.isInteger(id) || id < 0 || id >= names.length) return false;
      this.remember();
      const item = {id, box, score: null, origin: 'manual', uid: `manual-${this.nextManual++}`};
      this.items.push(item);
      this.selected = item.uid;
      return true;
    }
    removeSelected() {
      if (!this.items.some(d => d.uid === this.selected)) return false;
      this.remember();
      this.items = this.items.filter(d => d.uid !== this.selected);
      this.selected = null;
      return true;
    }
    undo() {
      const previous = this.history.pop();
      if (!previous) return false;
      this.items = previous.items;
      this.selected = previous.selected;
      return true;
    }
    loadReview(validated) {
      this.items = clone(validated.items);
      this.original = clone(validated.original);
      this.history = [];
      this.selected = null;
      this.nextManual = this.items.reduce((max,d) => d.origin === 'manual' ? Math.max(max,Number(d.uid.split('-')[1])+1) : max,1);
    }
    get hasCorrections() {
      return this.items.some(d => d.origin === 'manual')
        || this.items.filter(d => d.origin === 'model').length !== this.original.length;
    }
    restore() {
      if (!this.hasCorrections) return false;
      this.remember();
      this.items = clone(this.original);
      this.selected = null;
      return true;
    }
    csv(cutoff) {
      return ['detection_id,class,confidence,x_min,y_min,x_max,y_max,origin',
        ...this.visible(cutoff).map(d => [d.uid, names[d.id], d.score === null ? '' : d.score.toFixed(4),
          ...d.box.map(v => v.toFixed(2)), d.origin].join(','))].join('\n') + '\n';
    }
    snapshot(cutoff) {
      const remaining = new Set(this.items.map(d => d.uid));
      return {detections: clone(this.visible(cutoff)), all_detections: clone(this.items),
        original_predictions: clone(this.original), removed_predictions: clone(this.original.filter(d => !remaining.has(d.uid)))};
    }
  }

  const api = {ReviewStore, imagePoint, normalizedBox, hitTest, validateReview, names};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReviewTools = api;
})(globalThis);
