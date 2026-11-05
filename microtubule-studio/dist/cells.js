(function (root) {
  "use strict";
  function quantile(values, q) {
    const a = Array.from(values).sort((a, b) => a - b);
    return (
      a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))] ??
      0
    );
  }
  function hull(points) {
    const a = points
      .map((p) => [...p])
      .sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    const cross = (o, p, q) =>
      (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]);
    const lower = [],
      upper = [];
    for (const p of a) {
      while (lower.length > 1 && cross(lower.at(-2), lower.at(-1), p) <= 0)
        lower.pop();
      lower.push(p);
    }
    for (const p of a.reverse()) {
      while (upper.length > 1 && cross(upper.at(-2), upper.at(-1), p) <= 0)
        upper.pop();
      upper.push(p);
    }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }
  function inside(p, poly) {
    let result = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const a = poly[i],
        b = poly[j];
      if (
        a[1] > p[1] !== b[1] > p[1] &&
        p[0] < ((b[0] - a[0]) * (p[1] - a[1])) / (b[1] - a[1]) + a[0]
      )
        result = !result;
    }
    return result;
  }
  function area(poly) {
    return (
      Math.abs(
        poly.reduce((s, p, i) => {
          const q = poly[(i + 1) % poly.length];
          return s + p[0] * q[1] - q[0] * p[1];
        }, 0),
      ) / 2
    );
  }
  function validPolygon(poly) {
    if (
      !Array.isArray(poly) ||
      poly.length < 3 ||
      poly.length > 1000 ||
      !poly.every(
        (p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite),
      ) ||
      area(poly) < 16
    )
      return false;
    const cross = (a, b, c) =>
      (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const on = (a, b, p) =>
      Math.abs(cross(a, b, p)) < 1e-8 &&
      p[0] >= Math.min(a[0], b[0]) &&
      p[0] <= Math.max(a[0], b[0]) &&
      p[1] >= Math.min(a[1], b[1]) &&
      p[1] <= Math.max(a[1], b[1]);
    const intersect = (a, b, c, d) =>
      (cross(a, b, c) * cross(a, b, d) < 0 &&
        cross(c, d, a) * cross(c, d, b) < 0) ||
      on(a, b, c) ||
      on(a, b, d) ||
      on(c, d, a) ||
      on(c, d, b);
    for (let i = 0; i < poly.length; i++)
      for (let j = i + 1; j < poly.length; j++) {
        if (j === i + 1 || (i === 0 && j === poly.length - 1)) continue;
        if (
          intersect(
            poly[i],
            poly[(i + 1) % poly.length],
            poly[j],
            poly[(j + 1) % poly.length],
          )
        )
          return false;
      }
    return true;
  }
  function outline(gray, width, height, box, support = 0.7) {
    const step = Math.max(2, Math.ceil(Math.max(width, height) / 256));
    const samples = [];
    for (let y = Math.ceil(box[1] / step) * step; y < box[3]; y += step)
      for (let x = Math.ceil(box[0] / step) * step; x < box[2]; x += step) {
        let sum = 0,
          n = 0;
        for (let yy = y; yy < Math.min(y + step, height, box[3]); yy++)
          for (let xx = x; xx < Math.min(x + step, width, box[2]); xx++) {
            sum += gray[yy * width + xx];
            n++;
          }
        samples.push({
          p: [Math.min(width, x + step / 2), Math.min(height, y + step / 2)],
          value: sum / Math.max(n, 1),
        });
      }
    const cutoff = quantile(
      samples.map((s) => s.value),
      support,
    );
    const points = samples.filter((s) => s.value > cutoff).map((s) => s.p);
    let polygon = hull(points);
    const fallback = polygon.length < 3 || area(polygon) < 64;
    if (fallback)
      polygon = [
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]],
      ];
    return { points: polygon, fallback };
  }
  function preprocess(rgba, mode) {
    const out = new Uint8ClampedArray(rgba);
    if (mode === "original") return out;
    if (mode !== "percentile") throw Error("Unsupported preprocessing");
    const green = new Uint8Array(rgba.length / 4);
    const hist = new Uint32Array(256);
    for (let i = 0; i < green.length; i++) {
      green[i] = rgba[4 * i + 1];
      hist[green[i]]++;
    }
    function percentile(q) {
      const target = q * (green.length - 1),
        lower = Math.floor(target),
        upper = Math.ceil(target);
      let n = 0,
        lo = 0,
        hi = 255,
        found = false;
      for (let i = 0; i < 256; i++) {
        n += hist[i];
        if (!found && n > lower) {
          lo = i;
          found = true;
        }
        if (n > upper) {
          hi = i;
          break;
        }
      }
      return lo + (hi - lo) * (target - lower);
    }
    const lo = percentile(0.005),
      hi = percentile(0.995);
    for (let i = 0; i < green.length; i++) {
      const scaled = Math.max(
          0,
          Math.min(255, ((green[i] - lo) * 255) / Math.max(1, hi - lo)),
        ),
        floor = Math.floor(scaled);
      const value =
        scaled - floor === 0.5 ? floor + (floor % 2) : Math.round(scaled);
      for (let c = 0; c < 3; c++) out[4 * i + c] = c === 1 ? value : 0;
    }
    return out;
  }
  function ownership(box, cells) {
    const [x1, y1, x2, y2] = box,
      center = [(x1 + x2) / 2, (y1 + y2) / 2];
    const fractions = cells.map((cell) => {
      let hits = 0;
      for (let y = 0; y < 8; y++)
        for (let x = 0; x < 8; x++)
          hits += Number(
            inside(
              [
                x1 + ((x + 0.5) * (x2 - x1)) / 8,
                y1 + ((y + 0.5) * (y2 - y1)) / 8,
              ],
              cell.points,
            ),
          );
      return hits / 64;
    });
    const candidates = cells.map((c, i) => ({
      cell: c,
      fraction: fractions[i],
      center: inside(center, c.points),
    }));
    const centered = candidates.filter((c) => c.center),
      sorted = [...candidates].sort((a, b) => b.fraction - a.fraction);
    if (
      centered.length === 1 &&
      sorted[0]?.cell === centered[0].cell &&
      sorted[0].fraction >= 0.8 &&
      (sorted[1]?.fraction ?? 0) <= 0.1
    )
      return { cellId: sorted[0].cell.id, status: "assigned" };
    return {
      cellId: null,
      status: candidates.some((c) => c.center || c.fraction > 0)
        ? "boundary"
        : "outside",
    };
  }
  function counts(items, cells, cutoff) {
    const rows = cells.map((cell) => ({
        cell_id: cell.id,
        cytosolic_count: 0,
        model_count: 0,
        manual_count: 0,
      })),
      lookup = new Map(rows.map((r) => [r.cell_id, r]));
    const details = [];
    let boundary = 0,
      outside = 0;
    for (const d of items.filter(
      (d) => d.id === 0 && (d.origin === "manual" || d.score >= cutoff),
    )) {
      let assignment = ownership(d.box, cells);
      if (d.ownerOverride && lookup.has(d.ownerOverride))
        assignment = { cellId: d.ownerOverride, status: "manual_assignment" };
      const row = lookup.get(assignment.cellId);
      if (row) {
        row.cytosolic_count++;
        row[d.origin === "manual" ? "manual_count" : "model_count"]++;
      } else if (assignment.status === "boundary") boundary++;
      else outside++;
      details.push({
        ...d,
        cell_id: assignment.cellId,
        ownership_status: assignment.status,
      });
    }
    return { rows, details, boundary, outside };
  }
  const api = {
    quantile,
    hull,
    inside,
    area,
    validPolygon,
    outline,
    preprocess,
    ownership,
    counts,
  };
  if (typeof module !== "undefined") module.exports = api;
  root.CellWorkflow = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
