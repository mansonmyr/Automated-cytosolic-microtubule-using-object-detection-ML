(function (root) {
  "use strict";
  class MinHeap {
    constructor() {
      this.items = [];
      this.next = 0;
    }
    push(index, cost, label) {
      const item = { index, cost, label, order: this.next++ };
      const items = this.items;
      items.push(item);
      let position = items.length - 1;
      while (position > 0) {
        const parent = (position - 1) >> 1;
        if (!this.before(item, items[parent])) break;
        items[position] = items[parent];
        position = parent;
      }
      items[position] = item;
    }
    before(a, b) {
      return a.cost < b.cost || (a.cost === b.cost && a.order < b.order);
    }
    pop() {
      const items = this.items,
        first = items[0],
        last = items.pop();
      if (!items.length) return first;
      let position = 0;
      while (position * 2 + 1 < items.length) {
        let child = position * 2 + 1;
        if (
          child + 1 < items.length &&
          this.before(items[child + 1], items[child])
        )
          child++;
        if (!this.before(items[child], last)) break;
        items[position] = items[child];
        position = child;
      }
      items[position] = last;
      return first;
    }
  }

  function modelInput(rgba, width, height, size = 256) {
    const plane = size * size;
    const hist = new Uint32Array(256);
    for (let i = 0; i < width * height; i++) hist[rgba[4 * i + 1]]++;
    function percentile(q) {
      const position = q * (width * height - 1),
        lower = Math.floor(position),
        upper = Math.ceil(position);
      let total = 0,
        low = 0,
        high = 255,
        found = false;
      for (let value = 0; value < 256; value++) {
        total += hist[value];
        if (!found && total > lower) {
          low = value;
          found = true;
        }
        if (total > upper) {
          high = value;
          break;
        }
      }
      return low + (high - low) * (position - lower);
    }
    const low = percentile(0.005),
      high = percentile(0.995);
    const normalized = new Uint8Array(width * height);
    for (let i = 0; i < normalized.length; i++) {
      const value = Math.max(
        0,
        Math.min(
          255,
          ((rgba[4 * i + 1] - low) * 255) / Math.max(1, high - low),
        ),
      );
      const floor = Math.floor(value);
      normalized[i] =
        value - floor === 0.5 ? floor + (floor % 2) : Math.round(value);
    }
    const output = new Float32Array(3 * plane),
      means = [0.485, 0.456, 0.406],
      stds = [0.229, 0.224, 0.225];
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const x1 = (x * width) / size,
          x2 = ((x + 1) * width) / size;
        const y1 = (y * height) / size,
          y2 = ((y + 1) * height) / size;
        let sum = 0;
        for (let yy = Math.floor(y1); yy < Math.ceil(y2); yy++) {
          const wy = Math.min(y2, yy + 1) - Math.max(y1, yy);
          for (let xx = Math.floor(x1); xx < Math.ceil(x2); xx++) {
            const wx = Math.min(x2, xx + 1) - Math.max(x1, xx);
            sum +=
              normalized[
                Math.min(height - 1, yy) * width + Math.min(width - 1, xx)
              ] *
              wx *
              wy;
          }
        }
        const gray = Math.fround(sum / ((x2 - x1) * (y2 - y1)) / 255);
        for (let c = 0; c < 3; c++)
          output[c * plane + y * size + x] = Math.fround(
            Math.fround(gray - Math.fround(means[c])) / Math.fround(stds[c]),
          );
      }
    return output;
  }


  // Exact square-window argmax, with the original smallest-index tie rule.
  function localMaxima(energy, mask, size, radius) {
    const horizontal = new Int32Array(size * size).fill(-1);
    const output = new Int32Array(size * size).fill(-1);
    const better = (a, b) => b < 0 || (a >= 0 &&
      (energy[a] > energy[b] || (energy[a] === energy[b] && a < b)));
    function pass(vertical) {
      for (let line = 0; line < size; line++) {
        const positions = new Int32Array(size), values = new Int32Array(size);
        let head = 0, tail = 0, next = 0;
        for (let center = 0; center < size; center++) {
          const end = Math.min(size - 1, center + radius);
          while (next <= end) {
            const idx = vertical ? next * size + line : line * size + next;
            const value = vertical ? horizontal[idx] : (mask[idx] ? idx : -1);
            if (value >= 0) {
              while (tail > head && better(value, values[tail - 1])) tail--;
              positions[tail] = next; values[tail++] = value;
            }
            next++;
          }
          while (tail > head && positions[head] < center - radius) head++;
          const idx = vertical ? center * size + line : line * size + center;
          (vertical ? output : horizontal)[idx] = tail > head ? values[head] : -1;
        }
      }
    }
    pass(false); pass(true); return output;
  }

  function labelInstances(probabilities, size, options) {
    const plane = size * size;
    const foreground = probabilities.subarray(0, plane),
      boundary = probabilities.subarray(plane, 2 * plane),
      distance = probabilities.subarray(2 * plane);
    const mask = Uint8Array.from(foreground, (v) =>
      v >= options.foregroundThreshold ? 1 : 0,
    );
    const energy = Float32Array.from(
      distance,
      (v, i) => v - options.boundaryWeight * boundary[i],
    );
    const radius = options.seedDistance;
    const candidates = [];
    if (options.useCentrosomalMarkers) {
      for (const marker of [...(options.gridMarkers ?? [])].sort(
        (a, b) => b.score - a.score,
      )) {
        const cx = Math.round(marker.point[0]),
          cy = Math.round(marker.point[1]);
        let best = -1,
          bestValue = -Infinity;
        for (
          let y = Math.max(0, cy - 6);
          y <= Math.min(size - 1, cy + 6);
          y++
        ) {
          for (
            let x = Math.max(0, cx - 6);
            x <= Math.min(size - 1, cx + 6);
            x++
          ) {
            const index = y * size + x;
            const value = energy[index] - 0.025 * Math.hypot(x - cx, y - cy);
            if (mask[index] && value > bestValue) {
              best = index;
              bestValue = value;
            }
          }
        }
        if (
          best >= 0 &&
          !candidates.some(
            (index) =>
              Math.hypot(
                (index % size) - (best % size),
                Math.floor(index / size) - Math.floor(best / size),
              ) < options.markerDistance,
          )
        )
          candidates.push(best);
      }
    }
    const maxima = options.useCentrosomalMarkers ? null : localMaxima(energy, mask, size, radius);
    if (!options.useCentrosomalMarkers)
      for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++) {
          const index = y * size + x;
          if (!mask[index] || distance[index] < 0.2) continue;
          if (maxima[index] === index) candidates.push(index);
        }
    // Guarantee one marker in every foreground component.
    const visited = new Uint8Array(plane),
      queue = new Int32Array(plane),
      peaks = new Set(candidates);
    const neighbors = (index) => {
      const x = index % size,
        y = Math.floor(index / size),
        result = [];
      if (x > 0) result.push(index - 1);
      if (x + 1 < size) result.push(index + 1);
      if (y > 0) result.push(index - size);
      if (y + 1 < size) result.push(index + size);
      return result;
    };
    for (let start = 0; start < plane; start++) {
      if (!mask[start] || visited[start]) continue;
      let head = 0,
        tail = 1,
        best = start,
        hasPeak = false;
      queue[0] = start;
      visited[start] = 1;
      while (head < tail) {
        const index = queue[head++];
        if (energy[index] > energy[best]) best = index;
        if (peaks.has(index)) hasPeak = true;
        for (const next of neighbors(index))
          if (mask[next] && !visited[next]) {
            visited[next] = 1;
            queue[tail++] = next;
          }
      }
      if (!hasPeak && tail >= options.minimumArea) candidates.push(best);
    }
    const labels = new Int32Array(plane),
      heap = new MinHeap();
    candidates.forEach((index, i) => {
      labels[index] = i + 1;
      heap.push(index, -energy[index], i + 1);
    });
    while (heap.items.length) {
      const current = heap.pop();
      for (const next of neighbors(current.index)) {
        if (!mask[next] || labels[next]) continue;
        labels[next] = current.label;
        heap.push(next, Math.max(current.cost, -energy[next]), current.label);
      }
    }
    const areas = new Uint32Array(candidates.length + 1);
    for (const label of labels) areas[label]++;
    for (let i = 0; i < labels.length; i++)
      if (areas[labels[i]] < options.minimumArea) labels[i] = 0;
    return { labels, areas };
  }

  function simplify(points, tolerance) {
    if (points.length <= 3) return points;
    function openLine(line) {
      if (line.length <= 2) return line;
      const first = line[0],
        last = line.at(-1),
        dx = last[0] - first[0],
        dy = last[1] - first[1];
      let maximum = 0,
        split = 0;
      for (let i = 1; i < line.length - 1; i++) {
        const p = line[i],
          t = Math.max(
            0,
            Math.min(
              1,
              ((p[0] - first[0]) * dx + (p[1] - first[1]) * dy) /
                Math.max(dx * dx + dy * dy, 1e-12),
            ),
          );
        const distance = Math.hypot(
          p[0] - first[0] - t * dx,
          p[1] - first[1] - t * dy,
        );
        if (distance > maximum) {
          maximum = distance;
          split = i;
        }
      }
      if (maximum <= tolerance) return [first, last];
      return openLine(line.slice(0, split + 1))
        .slice(0, -1)
        .concat(openLine(line.slice(split)));
    }
    const halfway = Math.floor(points.length / 2);
    return openLine(points.slice(0, halfway + 1))
      .slice(0, -1)
      .concat(openLine(points.slice(halfway).concat([points[0]])).slice(0, -1));
  }

  function polygonsFromLabels(labels, size, width, height) {
    const edges = new Map();
    function add(label, a, b) {
      if (!edges.has(label)) edges.set(label, new Map());
      const map = edges.get(label),
        key = a[1] * (size + 1) + a[0];
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(b);
    }
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const label = labels[y * size + x];
        if (!label) continue;
        if (y === 0 || labels[(y - 1) * size + x] !== label)
          add(label, [x, y], [x + 1, y]);
        if (x === size - 1 || labels[y * size + x + 1] !== label)
          add(label, [x + 1, y], [x + 1, y + 1]);
        if (y === size - 1 || labels[(y + 1) * size + x] !== label)
          add(label, [x + 1, y + 1], [x, y + 1]);
        if (x === 0 || labels[y * size + x - 1] !== label)
          add(label, [x, y + 1], [x, y]);
      }
    const polygons = [];
    for (const [label, map] of edges) {
      const loops = [];
      while (map.size) {
        const first = map.keys().next().value;
        let point = [first % (size + 1), Math.floor(first / (size + 1))],
          loop = [];
        for (let count = 0; count <= size * size * 4; count++) {
          loop.push(point);
          const key = point[1] * (size + 1) + point[0],
            next = map.get(key);
          if (!next?.length) break;
          point = next.pop();
          if (!next.length) map.delete(key);
          if (point[1] * (size + 1) + point[0] === first) break;
        }
        if (loop.length >= 3) loops.push(loop);
      }
      const signedArea = (p) =>
        Math.abs(
          p.reduce((s, q, i) => {
            const r = p[(i + 1) % p.length];
            return s + q[0] * r[1] - r[0] * q[1];
          }, 0),
        );
      loops.sort((a, b) => signedArea(b) - signedArea(a));
      if (!loops.length) continue;
      const points = simplify(loops[0], 0.65).map((p) => [
        (p[0] * width) / size,
        (p[1] * height) / size,
      ]);
      if (points.length >= 3) polygons.push({ label, points });
    }
    return polygons;
  }

  // Group only markers in the same preliminary distance-based cell region.
  // Each seed represents one or two detections; a pair never forces two cells.
  function groupCentrosomes(markers, labels, size, pairDistance) {
    const groups = new Map();
    for (const marker of [...markers].sort((a,b) => b.score-a.score)) {
      const x=Math.round(marker.point[0]), y=Math.round(marker.point[1]);
      if(x<0 || y<0 || x>=size || y>=size) continue;
      const label=labels[y*size+x];
      if(!label) continue;
      if(!groups.has(label)) groups.set(label,[]);
      groups.get(label).push(marker);
    }
    const seeds=[];
    for(const group of groups.values()) {
      const remaining=[...group];
      while(remaining.length) {
        const first=remaining.shift();
        let nearest=-1, distance=pairDistance;
        remaining.forEach((m,i) => {
          const d=Math.hypot(m.point[0]-first.point[0],m.point[1]-first.point[1]);
          if(d<distance) {nearest=i;distance=d;}
        });
        const members=[first];
        if(nearest>=0) members.push(remaining.splice(nearest,1)[0]);
        seeds.push({...first, centrosomeCount:members.length});
      }
    }
    return seeds;
  }

  function pointInside(point, poly) {
    let yes=false;
    for(let i=0,j=poly.length-1;i<poly.length;j=i++) {
      const a=poly[i],b=poly[j];
      if((a[1]>point[1]) !== (b[1]>point[1]) && point[0] < (b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0]) yes=!yes;
    }
    return yes;
  }
  function filterEligible(polygons, width, height, options) {
    return polygons.filter(p => {
      const margin=options.borderMarginPixels ?? 2;
      if(options.excludeBorderCells && p.points.some(([x,y]) => x<=margin || y<=margin || x>=width-margin || y>=height-margin)) return false;
      if(options.requireCentrosomes) {
        const count=(options.markers ?? []).filter(m => m.score >= (options.markerThreshold ?? 0.25) && pointInside(m.point,p.points)).length;
        if(count<1 || count>2) return false;
      }
      return true;
    });
  }

  function segment(probabilities, size, width, height, options) {
    let gridMarkers = (options.markers ?? [])
      .filter((m) => m.score >= options.markerThreshold)
      .map((m) => ({
        score: m.score,
        point: [(m.point[0] * size) / width, (m.point[1] * size) / height],
      }));
    if(options.useCentrosomalMarkers && options.maxCentrosomesPerCell === 2) {
      const preliminary=labelInstances(probabilities,size,{...options,useCentrosomalMarkers:false});
      gridMarkers=groupCentrosomes(gridMarkers,preliminary.labels,size,options.centrosomePairDistance ?? 120);
    }
    const { labels } = labelInstances(probabilities, size, {
      ...options,
      gridMarkers,
    });
    const polygons = polygonsFromLabels(labels, size, width, height);
    return filterEligible(polygons, width, height, options);

  }
  const api = { filterEligible, groupCentrosomes, modelInput, labelInstances, polygonsFromLabels, segment };
  if (typeof module !== "undefined") module.exports = api;
  root.LearnedSegmentation = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
