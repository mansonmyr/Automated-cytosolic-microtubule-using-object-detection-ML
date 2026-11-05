"use strict";
const $ = (id) => document.getElementById(id),
  C = CellWorkflow;
const store = new ReviewTools.ReviewStore();
let excludedCells = [];
let eligibilityExamples = null;
let autoFit = true;
let cellSession = null;
let cellConfig = null;
let cellReviewModel = null;
const MODEL_SHA =
  "4f3f06f514a87daed9fc276bb1f7d678d044086661acabc278be00b0de0ea209";
let source = null,
  rgba = null,
  prepared = null,
  imageHash = "",
  fileName = "image",
  session = null;
let cells = [],
  cellHistory = [],
  selectedCell = null,
  nextCell = 1,
  approved = false,
  counted = false,
  busy = false,
  dirty = false;
let stage = 1,
  zoom = 1,
  tool = "select",
  draft = [],
  replaceId = null,
  drag = null,
  cache = new Map(),
  countPolicy = "original";
const clone = (x) => JSON.parse(JSON.stringify(x)),
  cutoff = () => Number($("confidence").value);
const cellName = (id) => "Cell " + id.split("-")[1];
const say = (text) => {
  $("status").textContent = text;
};
function button(text, action, selected = false) {
  const b = document.createElement("button");
  b.textContent = text;
  b.onclick = action;
  if (selected) b.className = "selected";
  return b;
}
function invalidateCells() {
  approved = false;
  dirty = true;
  $("reviewed").checked = false;
  const ids = new Set(cells.map((c) => c.id));
  store.items.forEach((d) => {
    if (d.ownerOverride && !ids.has(d.ownerOverride)) delete d.ownerOverride;
  });
}
function rememberCells() {
  const snapshot = clone(cells);
  snapshot.excludedSnapshot = clone(excludedCells);
  cellHistory.push(snapshot);
  if (cellHistory.length > 50) cellHistory.shift();
}
function resetDraft() {
  draft = [];
  replaceId = null;
  drag = null;
  tool = "select";
}
function setStage(value) {
  if (busy || (!source && value > 1) || (value > 2 && !approved)) return;
  if (draft.length) {
    say("Finish or cancel the polygon before changing stage.");
    return;
  }
  stage = value;
  resetDraft();
  render();
}
function refreshPrepared() {
  if (!source) return;
  const mode = $("preprocessing").value;
  if (prepared?.mode === mode) return;
  const canvas = document.createElement("canvas");
  canvas.width = source.width;
  canvas.height = source.height;
  canvas
    .getContext("2d")
    .putImageData(
      new ImageData(C.preprocess(rgba, mode === "flip-agreement" ? "original" : mode), source.width, source.height),
      0,
      0,
    );
  prepared = { mode, canvas };
}
function paint() {
  if (!source) return;
  const canvas = $("canvas"),
    ctx = canvas.getContext("2d");
  const display =
    stage >= 3 && !$("show-original").checked
      ? (prepared?.canvas ?? source)
      : source;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(display, 0, 0);
  const line = Math.max(1, 2 / zoom),
    font = Math.max(11, 12 / zoom);
  for (const cell of cells) {
    ctx.beginPath();
    cell.points.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
    ctx.closePath();
    ctx.fillStyle = cell.id === selectedCell ? "#89adff22" : "#89adff0a";
    ctx.fill();
    ctx.strokeStyle = cell.id === selectedCell ? "#ffafd8" : "#8dabff";
    ctx.lineWidth = line;
    ctx.stroke();
    ctx.font = `${font}px system-ui`;
    ctx.fillStyle = "#fff";
    ctx.fillText(
      cellName(cell.id),
      cell.points[0][0] + 6 / zoom,
      cell.points[0][1] - 6 / zoom,
    );
    if (stage === 2 && cell.id === selectedCell)
      for (const p of cell.points) {
        ctx.fillStyle = "#ffafd8";
        ctx.fillRect(p[0] - 4 / zoom, p[1] - 4 / zoom, 8 / zoom, 8 / zoom);
      }
  }
  if (stage >= 2 && counted) {
    const results = C.counts(store.items, cells, cutoff());
    for (const d of results.details) {
      ctx.strokeStyle =
        d.ownership_status === "boundary"
          ? "#ffc868"
          : d.ownership_status === "outside"
            ? "#8c9ab3"
            : "#ffa6d6";
      ctx.lineWidth = d.uid === store.selected ? line * 2 : line;
      ctx.setLineDash(d.origin === "manual" ? [5 / zoom, 3 / zoom] : []);
      ctx.strokeRect(
        d.box[0],
        d.box[1],
        d.box[2] - d.box[0],
        d.box[3] - d.box[1],
      );
    }
    ctx.setLineDash([]);
  }
  if (draft.length) {
    ctx.beginPath();
    draft.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
    ctx.strokeStyle = "#ffadd6";
    ctx.lineWidth = line;
    ctx.stroke();
    for (const p of draft) {
      ctx.fillStyle = "#ffadd6";
      ctx.fillRect(p[0] - 3 / zoom, p[1] - 3 / zoom, 6 / zoom, 6 / zoom);
    }
  }
  if (drag?.kind === "box") {
    ctx.strokeStyle = "#fff";
    ctx.strokeRect(
      drag.start[0],
      drag.start[1],
      drag.current[0] - drag.start[0],
      drag.current[1] - drag.start[1],
    );
  }
}
function render() {
  document
    .querySelectorAll("[data-panel]")
    .forEach((el) => (el.hidden = Number(el.dataset.panel) !== stage));
  document.querySelectorAll("[data-stage]").forEach((el) => {
    el.setAttribute(
      "aria-current",
      Number(el.dataset.stage) === stage ? "step" : "false",
    );
    el.disabled =
      busy ||
      (!source && Number(el.dataset.stage) > 1) ||
      (Number(el.dataset.stage) > 2 && !approved);
  });
  $("stage-title").textContent = [
    "",
    "Detect cells",
    "Review cell polygons",
    "Choose preprocessing",
    "Count cytosolic objects",
  ][stage];
  for (const id of [
    "file",
    "cell-confidence",
    "preprocessing",
    "show-original",
    "display-brightness",
    "auto-exclude",
    "confidence",
  ])
    $(id).disabled = busy;
  for (const id of [
    "auto-cells",
    "manual-start",
    "load-review",
    "zoom-in",
    "zoom-out",
    "zoom-fit",
    "export-png",
    "save-review",
  ])
    $(id).disabled = busy || !source;
  for (const id of ["select-tool", "polygon-tool", "box-select", "box-draw"])
    $(id).disabled = busy || !source;
  $("finish-polygon").disabled = busy || draft.length < 3;
  $("cancel-polygon").disabled = busy || tool !== "polygon";
  $("replace-polygon").disabled = $("delete-cell").disabled =
    busy || !selectedCell;
  $("undo-cell").disabled = busy || !cellHistory.length;
  $("approve-cells").disabled =
    busy ||
    !cells.length ||
    !!draft.length ||
    !cells.every((c) => C.validPolygon(c.points));
  $("run-count").disabled = busy || !approved;
  $("to-count").disabled = busy || !approved;
  $("delete-box").disabled = busy || !store.selected;
  $("undo-box").disabled = busy || !store.history.length;
  for (const id of ["export-counts", "export-boxes", "reviewed"])
    $(id).disabled = busy || !counted || !approved;
  $("cell-list").replaceChildren(
    ...cells.map((cell) =>
      button(
        cellName(cell.id) + (cell.fallback ? " · box fallback" : ""),
        () => {
          selectedCell = cell.id;
          render();
        },
        selectedCell === cell.id,
      ),
    ),
  );
  $("cell-selection").textContent = selectedCell
    ? `${cellName(selectedCell)} selected. Drag a vertex or redraw its boundary.`
    : "Select a polygon.";
  const selected = store.items.find((d) => d.uid === store.selected);
  $("assign-cell").replaceChildren(
    new Option("Automatic ownership", ""),
    ...cells.map((c) => new Option(cellName(c.id), c.id)),
  );
  $("assign-cell").value = selected?.ownerOverride ?? "";
  $("assign-cell").disabled = busy || !selected;
  const result = C.counts(store.items, cells, cutoff());
  $("count-results").hidden = !counted;
  $("restore-excluded").disabled = busy || !excludedCells.length;
  $("exclusion-info").textContent = `${excludedCells.length} proposals excluded by example similarity. Restore them to inspect or correct this experimental filter.`;
  $("confidence-balanced").disabled = busy;
  $("confidence-precision").disabled = busy;
  $("confidence-live").textContent = `Cutoff ${cutoff().toFixed(3)}: ${result.details.length} visible cytosolic candidates. Lower cutoffs show more candidates, including more false detections.`;
  $("count-rows").replaceChildren(
    ...result.rows.map((row) => {
      const tr = document.createElement("tr");
      for (const value of [
        cellName(row.cell_id),
        row.cytosolic_count,
        row.model_count,
        row.manual_count,
      ]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      return tr;
    }),
  );
  $("unresolved").textContent =
    `${result.boundary} boundary candidates need assignment. ${result.outside} candidates are outside all cell polygons.`;
  $("result-state").textContent = $("reviewed").checked
    ? "Detections checked"
    : approved ? "Model candidates" : "Preview · full-cell confirmation required";
  const items = result.details;
  $("detection-list").replaceChildren(
    ...items.slice(0, 250).map((d) =>
      button(
        `${d.uid} · ${d.cell_id ? cellName(d.cell_id) : d.ownership_status} · ${d.score === null ? "manual" : d.score.toFixed(3)}`,
        () => {
          store.selected = d.uid;
          render();
        },
        d.uid === store.selected,
      ),
    ),
  );
  if (items.length > 250) {
    const p = document.createElement("p");
    p.textContent = `Showing the first 250 of ${items.length} boxes. Select other boxes on the image.`;
    $("detection-list").append(p);
  }
  $("help").textContent =
    stage === 2
      ? "Select a cell, then drag a vertex. Draw polygons with clicks and press Enter to finish. Press Escape to cancel."
      : stage === 4
        ? "Select a cytosolic box to remove it or assign its cell. Drag with Add cytosolic box to mark a missed object."
        : "Scroll the viewer when zoomed in. Fit shows the entire image.";
  if (busy)
    document
      .querySelectorAll("button,input,select")
      .forEach((el) => (el.disabled = true));
  paint();
}
function fit() {
  if (!source) return;
  autoFit = true;
  zoom = Math.max(
    0.05,
    Math.min(
      1,
      ($("viewer").clientWidth - 10) / source.width,
      ($("viewer").clientHeight - 10) / source.height,
    ),
  );
  applyZoom();
  $("viewer").scrollLeft = 0;
  $("viewer").scrollTop = 0;
}
function applyZoom() {
  if (!source) return;
  $("canvas").style.width = source.width * zoom + "px";
  $("canvas").style.height = source.height * zoom + "px";
  paint();
}
function point(event) {
  return ReviewTools.imagePoint(
    event.clientX,
    event.clientY,
    $("canvas").getBoundingClientRect(),
    source.width,
    source.height,
  );
}
async function inference(mode) {
  if (cache.has(mode)) return clone(cache.get(mode));
  if (mode === "flip-agreement") {
    const original = (await inference("original")).filter(d => d.id === 0);
    const flipped = (await inference("flip-original")).filter(d => d.id === 0);
    const merged = mergeFlipAgreement(original, flipped).filter(d => d.score >= 0.05);
    cache.set(mode, clone(merged));
    return merged;
  }
  if (typeof ort === "undefined")
    throw Error(
      "Inference runtime did not load. Check the network connection.",
    );
  if (!session) {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("refined1024.onnx", {
      executionProviders: ["wasm"],
    });
  }
  const flip = mode === "flip-original";
  const data = C.preprocess(rgba, flip ? "original" : mode),
    input = wholeImageInput(data, source.width, source.height, 1024);
  if (flip) {
    // Reverse each row within each channel, including letterbox padding.
    for (let offset = 0; offset < input.input.length; offset += 1024)
      input.input.subarray(offset, offset + 1024).reverse();
  }
  const tensor = new ort.Tensor("float32", input.input, [1, 3, 1024, 1024]);
  let result;
  try {
    result = await session.run({ [session.inputNames[0]]: tensor });
    const output = result[session.outputNames[0]];
    if (output.dims.length !== 3 || output.dims[2] !== 6)
      throw Error("Unexpected model output.");
    const boxes = [];
    for (let i = 0; i < output.data.length; i += 6) {
      const a = output.data,
        score = a[i + 4],
        id = a[i + 5];
      if (
        !Number.isInteger(id) ||
        id < 0 ||
        id > 2 ||
        !Number.isFinite(score) ||
        score < 0.05 ||
        score > 1
      )
        continue;
      const coordinates = flip
        ? [1024 - a[i + 2], a[i + 1], 1024 - a[i], a[i + 3]]
        : [a[i], a[i + 1], a[i + 2], a[i + 3]];
      const box = input.mapBox(coordinates);
      if (box.every(Number.isFinite) && box[2] > box[0] && box[3] > box[1])
        boxes.push({ id, score, box });
    }
    const kept = suppressOverlaps(boxes);
    cache.set(mode, clone(kept));
    return kept;
  } finally {
    tensor.dispose();
    if (result) Object.values(result).forEach((t) => t.dispose());
  }
}
async function work(message, action) {
  busy = true;
  say(message);
  render();
  await new Promise((r) => setTimeout(r, 30));
  try {
    await action();
  } catch (error) {
    say(error.message);
  } finally {
    busy = false;
    render();
  }
}
$("file").onchange = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (
    dirty &&
    !confirm("Opening another image replaces this review. Continue?")
  ) {
    event.target.value = "";
    return;
  }
  await work("Opening image…", async () => {
    const buffer = await file.arrayBuffer(),
      canvas = document.createElement("canvas"),
      ctx = canvas.getContext("2d");
    if (/\.tiff?$/i.test(file.name)) {
      if (typeof UTIF === "undefined") throw Error("TIFF reader did not load.");
      const ifds = UTIF.decode(buffer);
      if (ifds.length !== 1) throw Error("Use a single-page TIFF.");
      UTIF.decodeImage(buffer, ifds[0]);
      canvas.width = ifds[0].width;
      canvas.height = ifds[0].height;
      ctx.putImageData(
        new ImageData(
          new Uint8ClampedArray(UTIF.toRGBA8(ifds[0])),
          canvas.width,
          canvas.height,
        ),
        0,
        0,
      );
    } else {
      const bitmap = await createImageBitmap(file);
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
    }
    if (
      !canvas.width ||
      canvas.width > 8192 ||
      !canvas.height ||
      canvas.height > 8192
    )
      throw Error("Use an image up to 8192 × 8192 pixels.");
    let hash;
    if (globalThis.crypto?.subtle)
      hash = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)),
        (x) => x.toString(16).padStart(2, "0"),
      ).join("");
    else if (typeof sha256 === "function") hash = sha256(buffer);
    else
      throw Error(
        "Image fingerprint library did not load. Use localhost or HTTPS.",
      );
    imageHash = hash;
    cellReviewModel = null;
    excludedCells = [];
    source = canvas;
    rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    fileName = file.name;
    cells = [];
    cellHistory = [];
    nextCell = 1;
    selectedCell = null;
    approved = false;
    counted = false;
    dirty = false;
    cache.clear();
    store.reset();
    prepared = null;
    stage = 1;
    resetDraft();
    $("reviewed").checked = false;
    $("show-original").checked = false;
    $("canvas").width = canvas.width;
    $("canvas").height = canvas.height;
    $("canvas").hidden = false;
    $("empty").hidden = true;
    $("image-info").textContent =
      `${fileName} · ${canvas.width} × ${canvas.height} px`;
    refreshPrepared();
    fit();
    say("Image loaded. Auto-detect the cell polygons to begin.");
  });
};
$("auto-cells").onclick = () => {
  if (
    cells.length &&
    !confirm(
      "Replace the current cell polygons with new segmentation proposals?",
    )
  )
    return;
  work("Loading the learned cell segmentation model…", async () => {
    if (typeof ort === "undefined")
      throw Error("Inference runtime did not load.");
    if (!cellConfig) {
      const response = await fetch("segmentation-config.json");
      if (!response.ok) throw Error("Cell segmentation settings did not load.");
      cellConfig = await response.json();
    }
    if (!cellSession) {
      ort.env.wasm.numThreads = 1;
      cellSession = await ort.InferenceSession.create(
        "cell_segmentation.onnx",
        { executionProviders: ["wasm"] },
      );
    }
    const input = LearnedSegmentation.modelInput(
      rgba,
      source.width,
      source.height,
      cellConfig.inputSize,
    );
    const tensor = new ort.Tensor("float32", input, [1, 3, cellConfig.inputSize, cellConfig.inputSize]);
    let results;
    try {
      results = await cellSession.run({ [cellSession.inputNames[0]]: tensor });
      const maps = results[cellSession.outputNames[0]];
      if (maps.dims.join(",") !== `1,3,${cellConfig.inputSize},${cellConfig.inputSize}`)
        throw Error("Unexpected segmentation output.");
      const settings = {
        ...cellConfig.postprocessing,
        foregroundThreshold: Number($("cell-confidence").value),
      };
      if (settings.useCentrosomalMarkers || settings.requireCentrosomes) {
        settings.markers = (await inference("original"))
          .filter((d) => d.id === 1)
          .map((d) => ({
            score: d.score,
            point: [(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2],
          }));
      }
      const proposals = LearnedSegmentation.segment(
        maps.data,
        cellConfig.inputSize,
        source.width,
        source.height,
        settings,
      );
      const valid = proposals.filter((p) => C.validPolygon(p.points));
      cellReviewModel = {
        name: cellConfig.name,
        onnx_sha256: cellConfig.onnx_sha256,
        foreground_threshold: Number($("cell-confidence").value),
      };
      rememberCells();
      cells = valid.map((p) => ({
        id: `cell-${nextCell++}`,
        points: p.points,
        fallback: false,
        origin: "automatic",
        segmentation_model: cellConfig.name,
      }));
      excludedCells = [];
      if ($("auto-exclude").checked) {
        if (!eligibilityExamples) {
          const response = await fetch("eligibility-examples.json");
          if (!response.ok) throw Error("Could not load exclusion examples.");
          eligibilityExamples = await response.json();
        }
        cells = cells.filter(cell => {
          const features = CellEligibility.describe(cell.points, rgba, source.width, source.height);
          const verdict = CellEligibility.classify(features, eligibilityExamples);
          if (verdict.exclude) { excludedCells.push({...cell, exclusion:verdict}); return false; }
          return true;
        });
      }
      selectedCell = cells[0]?.id ?? null;
      invalidateCells();
      stage = 2;
      resetDraft();
      say(
        `${cells.length} learned cell polygon proposals generated. Check boundaries before continuing.`,
      );
    } finally {
      tensor.dispose();
      if (results) Object.values(results).forEach((output) => output.dispose());
    }
    const policy = $("preprocessing").value;
    const predictions = (await inference(policy)).filter((d) => d.id === 0);
    store.reset(predictions);
    countPolicy = policy;
    counted = true;
    dirty = true;
    $("reviewed").checked = false;
    say(`${cells.length} cell proposals and ${predictions.length} microtubule candidates are ready. Remove partial or half cells, then confirm that every retained cell is complete before exporting counts.`);
  });
};
$("cell-confidence").oninput = () => {
  $("cell-threshold").value = Number($("cell-confidence").value).toFixed(3);
};
$("manual-start").onclick = () => {
  stage = 2;
  tool = "polygon";
  render();
  say("Click around the first cell and press Enter to finish.");
};
$("select-tool").onclick = () => {
  resetDraft();
  render();
};
$("polygon-tool").onclick = () => {
  resetDraft();
  tool = "polygon";
  render();
  say("Click around the cell and press Enter to finish.");
};
$("replace-polygon").onclick = () => {
  resetDraft();
  replaceId = selectedCell;
  tool = "polygon";
  render();
  say("Draw a replacement polygon and press Enter.");
};
function finishPolygon() {
  if (!C.validPolygon(draft)) {
    say(
      "The polygon needs at least three vertices, a visible area and no crossing edges.",
    );
    return;
  }
  rememberCells();
  const cell = {
    id: replaceId ?? `cell-${nextCell++}`,
    points: clone(draft),
    fallback: false,
    origin: "manual",
  };
  if (replaceId) cells = cells.map((c) => (c.id === replaceId ? cell : c));
  else cells.push(cell);
  selectedCell = cell.id;
  invalidateCells();
  resetDraft();
  render();
  say("Cell polygon saved.");
}
$("finish-polygon").onclick = finishPolygon;
$("cancel-polygon").onclick = () => {
  resetDraft();
  render();
};
$("delete-cell").onclick = () => {
  rememberCells();
  cells = cells.filter((c) => c.id !== selectedCell);
  selectedCell = null;
  invalidateCells();
  render();
};
$("undo-cell").onclick = () => {
  if (cellHistory.length) {
    cells = cellHistory.pop();
    if (cells.excludedSnapshot) { excludedCells = cells.excludedSnapshot; delete cells.excludedSnapshot; }
    selectedCell = null;
    invalidateCells();
    resetDraft();
    render();
  }
};
$("approve-cells").onclick = () => {
  if (!cells.length || !cells.every((c) => C.validPolygon(c.points))) return;
  approved = true;
  dirty = true;
  stage = 3;
  resetDraft();
  refreshPrepared();
  render();
  say("Retained cells confirmed as complete. Choose the input for cytosolic counting.");
};
$("preprocessing").onchange = () => {
  if ($("preprocessing").value === "flip-agreement") {
    $("confidence").value = "0.075";
    $("threshold").value = "0.075";
  }
  prepared = null;
  refreshPrepared();
  counted = false;
  $("reviewed").checked = false;
  dirty = true;
  render();
};
$("show-original").onchange = paint;
$("to-count").onclick = () => setStage(4);
$("run-count").onclick = () => {
  if (
    (store.hasCorrections || store.items.some((d) => d.ownerOverride)) &&
    !confirm("Running detection replaces the cytosolic corrections. Continue?")
  )
    return;
  work(
    "Detecting cytosolic candidates and assigning them to cells…",
    async () => {
      const policy = $("preprocessing").value;
      const predictions = (await inference(policy)).filter((d) => d.id === 0);
      store.reset(predictions);
      countPolicy = policy;
      counted = true;
      dirty = true;
      $("reviewed").checked = false;
      tool = "select";
      say(
        "Per-cell candidate counts are ready. Review boundary candidates and missed objects.",
      );
    },
  );
};
$("confidence").oninput = () => {
  $("threshold").value = cutoff().toFixed(3);
  $("reviewed").checked = false;
  dirty = true;
  render();
};
$("box-select").onclick = () => {
  tool = "select";
  render();
};
$("box-draw").onclick = () => {
  tool = "box";
  render();
};
$("delete-box").onclick = () => {
  store.removeSelected();
  dirty = true;
  $("reviewed").checked = false;
  render();
};
$("undo-box").onclick = () => {
  store.undo();
  dirty = true;
  $("reviewed").checked = false;
  render();
};
$("assign-cell").onchange = () => {
  const d = store.items.find((d) => d.uid === store.selected);
  if (d) {
    store.remember();
    d.ownerOverride = $("assign-cell").value || null;
    dirty = true;
    $("reviewed").checked = false;
    render();
  }
};
$("reviewed").onchange = () => {
  dirty = true;
  render();
};
$("canvas").onpointerdown = (event) => {
  if (busy || !source) return;
  $("viewer").focus();
  const p = point(event);
  if (stage === 2) {
    if (tool === "polygon") {
      draft.push(p);
      render();
      return;
    }
    const cell = cells.find((c) => c.id === selectedCell);
    const vertex =
      cell?.points.findIndex(
        (q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 10 / zoom,
      ) ?? -1;
    if (vertex >= 0) {
      drag = { kind: "vertex", cell, vertex, before: clone(cells) };
      $("canvas").setPointerCapture(event.pointerId);
      return;
    }
    const hits = cells
      .filter((c) => C.inside(p, c.points))
      .sort((a, b) => C.area(a.points) - C.area(b.points));
    selectedCell = hits[0]?.id ?? null;
    render();
  } else if (stage === 4 && counted) {
    if (tool === "box") {
      drag = { kind: "box", start: p, current: p };
      $("canvas").setPointerCapture(event.pointerId);
    } else {
      store.selected = ReviewTools.hitTest(p, store.visible(cutoff()));
      render();
    }
  }
};
$("canvas").onpointermove = (event) => {
  if (!drag) return;
  const p = point(event);
  if (drag.kind === "vertex") drag.cell.points[drag.vertex] = p;
  else drag.current = p;
  paint();
};
$("canvas").onpointerup = (event) => {
  if (!drag) return;
  if (drag.kind === "vertex") {
    if (C.validPolygon(drag.cell.points)) {
      cellHistory.push(drag.before);
      drag.cell.origin = "manual";
      invalidateCells();
    } else {
      cells = drag.before;
      say(
        "That edit would create an invalid polygon. The previous boundary was restored.",
      );
    }
  } else {
    store.add(0, drag.start, point(event), source.width, source.height);
    dirty = true;
    $("reviewed").checked = false;
  }
  drag = null;
  render();
};
$("canvas").onpointercancel = () => {
  if (drag?.kind === "vertex") cells = drag.before;
  drag = null;
  render();
};
$("viewer").onkeydown = (event) => {
  if (busy) return;
  if (event.key === "Enter" && stage === 2 && draft.length) {
    event.preventDefault();
    finishPolygon();
  }
  if (event.key === "Escape") {
    if (drag?.kind === "vertex") cells = drag.before;
    resetDraft();
    render();
  }
};
$("zoom-fit").onclick = fit;
$("zoom-in").onclick = () => {
  autoFit = false;
  zoom = Math.min(8, zoom * 1.3);
  applyZoom();
};
$("zoom-out").onclick = () => {
  autoFit = false;
  zoom = Math.max(0.05, zoom / 1.3);
  applyZoom();
};
document
  .querySelectorAll("[data-stage]")
  .forEach((b) => (b.onclick = () => setStage(Number(b.dataset.stage))));
function download(content, type, name) {
  const url = URL.createObjectURL(new Blob([content], { type })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const baseName = () => fileName.replace(/\.[^.]+$/, "");
function csvValue(v) {
  return '"' + String(v ?? "").replaceAll('"', '""') + '"';
}
$("export-counts").onclick = () => {
  const result = C.counts(store.items, cells, cutoff()),
    header = [
      "image",
      "cell_id",
      "cytosolic_count",
      "model_count",
      "manual_count",
      "unresolved_boundary_candidates",
      "outside_candidates",
      "preprocessing",
      "confidence",
      "review_status",
    ];
  const rows = result.rows.map((r) => [
    fileName,
    r.cell_id,
    r.cytosolic_count,
    r.model_count,
    r.manual_count,
    result.boundary,
    result.outside,
    countPolicy,
    cutoff(),
    $("reviewed").checked ? "detections_checked" : "model_candidates",
  ]);
  download(
    [header, ...rows].map((r) => r.map(csvValue).join(",")).join("\n") + "\n",
    "text/csv",
    baseName() + "-per-cell-counts.csv",
  );
};
$("export-boxes").onclick = () => {
  const result = C.counts(store.items, cells, cutoff()),
    rows = [
      [
        "detection_id",
        "cell_id",
        "ownership_status",
        "origin",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
      ],
      ...result.details.map((d) => [
        d.uid,
        d.cell_id,
        d.ownership_status,
        d.origin,
        d.score,
        ...d.box,
      ]),
    ];
  download(
    rows.map((r) => r.map(csvValue).join(",")).join("\n") + "\n",
    "text/csv",
    baseName() + "-detection-audit.csv",
  );
};
$("save-review").onclick = () => {
  const review = {
    schema: "per-cell-v1",
    auto_excluded_cells: clone(excludedCells),
    exclusion_method: "experimental reviewed-example similarity",
    schema_version: 2,
    source_file_sha256: imageHash,
    image_width: source.width,
    image_height: source.height,
    file: fileName,
    model_sha256: MODEL_SHA,
    model: {
      name: "Refined YOLOv10n 1024",
      url: "refined1024.onnx",
      inputSize: 1024,
      overlap: 0,
      mode: "whole_image",
      classes: ["cytosolic", "centrosomal", "cell"],
    },
    cells: clone(cells),
    cell_segmentation: cellReviewModel,
    boundaries_confirmed: approved,
    preprocessing: $("preprocessing").value,
    counted,
    count_policy: counted ? countPolicy : $("preprocessing").value,
    confidence_threshold: cutoff(),
    detections_checked: $("reviewed").checked,
    ...store.snapshot(cutoff()),
  };
  download(
    JSON.stringify(review, null, 2),
    "application/json",
    baseName() + "-per-cell-review.json",
  );
  dirty = false;
};
$("load-review").onclick = () => {
  $("review-file").value = "";
  $("review-file").click();
};
$("review-file").onchange = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  await work("Restoring review…", async () => {
    if (file.size > 10 * 1024 * 1024)
      throw Error("Review JSON must be smaller than 10 MB.");
    const r = JSON.parse(await file.text());
    if (
      r.schema !== "per-cell-v1" ||
      r.model_sha256 !== MODEL_SHA ||
      !["original", "percentile", "flip-agreement"].includes(r.preprocessing) ||
      r.count_policy !== r.preprocessing ||
      typeof r.counted !== "boolean" ||
      typeof r.boundaries_confirmed !== "boolean" ||
      typeof r.detections_checked !== "boolean"
    )
      throw Error("This file is not a supported per-cell review.");
    const validated = ReviewTools.validateReview(
      r,
      source.width,
      source.height,
      imageHash,
    );
    if (!Array.isArray(r.cells) || r.cells.length > 2000)
      throw Error("Invalid cell polygons.");
    const ids = new Set();
    const restored = r.cells.map((c) => {
      if (
        !/^cell-[1-9]\d{0,6}$/.test(c.id) ||
        ids.has(c.id) ||
        !C.validPolygon(c.points) ||
        !c.points.every(
          (p) =>
            p[0] >= 0 &&
            p[0] <= source.width &&
            p[1] >= 0 &&
            p[1] <= source.height,
        )
      )
        throw Error("A cell polygon is invalid.");
      ids.add(c.id);
      return {
        id: c.id,
        points: clone(c.points),
        origin: c.origin === "automatic" ? "automatic" : "manual",
        fallback: !!c.fallback,
      };
    });
    if ((r.boundaries_confirmed || r.counted) && !restored.length)
      throw Error("Confirmed results need cell polygons.");
    const overrides = new Map(
      r.all_detections
        .filter((d) => d.ownerOverride)
        .map((d) => [d.uid, d.ownerOverride]),
    );
    if ([...overrides.values()].some((id) => !ids.has(id)))
      throw Error("A saved cell assignment is invalid.");
    if (dirty && !confirm("Replace the current work with this saved review?"))
      return;
    cellReviewModel = r.cell_segmentation ? clone(r.cell_segmentation) : null;
    cells = restored;
    cellHistory = [];
    nextCell = Math.max(0, ...cells.map((c) => Number(c.id.split("-")[1]))) + 1;
    selectedCell = null;
    resetDraft();
    approved = r.boundaries_confirmed;
    counted = r.counted && approved;
    countPolicy = r.count_policy;
    store.loadReview(validated);
    store.items.forEach((d) => {
      if (overrides.has(d.uid)) d.ownerOverride = overrides.get(d.uid);
    });
    $("preprocessing").value = r.preprocessing;
    prepared = null;
    refreshPrepared();
    $("confidence").value = r.confidence_threshold;
    $("threshold").value = cutoff().toFixed(3);
    $("reviewed").checked = r.detections_checked && counted;
    stage = counted ? 4 : approved ? 3 : 2;
    dirty = false;
    say("Per-cell review restored for the matching original image.");
  });
};
$("export-png").onclick = () => {
  paint();
  $("canvas").toBlob((blob) => {
    if (!blob) {
      say("PNG export failed.");
      return;
    }
    download(blob, "image/png", baseName() + "-per-cell-review.png");
  });
};
window.addEventListener("beforeunload", (event) => {
  if (dirty) {
    event.preventDefault();
    event.returnValue = "";
  }
});
render();

// Display brightness never changes model inputs or exported counts.
$("display-brightness").oninput = () => {
  $("canvas").style.filter = `brightness(${$("display-brightness").value})`;
};
let resizeFrame;
new ResizeObserver(() => {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => { if (source && autoFit) fit(); });
}).observe($("viewer"));

$("restore-excluded").onclick = () => {
  rememberCells(); cells.push(...excludedCells); excludedCells=[]; invalidateCells(); render();
};
$("viewer").addEventListener("wheel", event => {
  if (!source || busy) return;
  event.preventDefault();
  const viewer=$("viewer"), rect=$("canvas").getBoundingClientRect();
  const px=(event.clientX-rect.left)/zoom, py=(event.clientY-rect.top)/zoom;
  const before=[rect.left+px*zoom,rect.top+py*zoom];
  autoFit=false; zoom=Math.max(.05,Math.min(8,zoom*Math.exp(-event.deltaY*.002)));applyZoom();
  const after=$("canvas").getBoundingClientRect();
  viewer.scrollLeft+=after.left+px*zoom-before[0];viewer.scrollTop+=after.top+py*zoom-before[1];
},{passive:false});
let pan=null;
$("viewer").addEventListener("pointerdown",event=>{
  if(!source || busy || !(event.button===1 || event.altKey))return;
  event.preventDefault();event.stopPropagation();autoFit=false;
  pan={x:event.clientX,y:event.clientY,left:$("viewer").scrollLeft,top:$("viewer").scrollTop};
  $("viewer").setPointerCapture(event.pointerId);
},true);
$("viewer").addEventListener("pointermove",event=>{
  if(!pan)return;event.stopPropagation();
  $("viewer").scrollLeft=pan.left+pan.x-event.clientX;$("viewer").scrollTop=pan.top+pan.y-event.clientY;
},true);
$("viewer").addEventListener("pointerup",event=>{if(pan){pan=null;event.stopPropagation();}},true);
$("viewer").addEventListener("pointercancel",()=>{pan=null;});
$("viewer").ondblclick=event=>{if(tool!=="polygon" && tool!=="box"){event.preventDefault();fit();}};

for (const [id,value] of [["confidence-balanced",0.05],["confidence-precision",0.174]]) {
  $(id).onclick=()=>{
    $("confidence").value=String(value);
    $("threshold").value=value.toFixed(3);
    $("reviewed").checked=false;dirty=true;render();
    say(`Cytosolic cutoff set to ${value.toFixed(3)}. Review the visible detections before export.`);
  };
}
