'use strict';
const $ = id => document.getElementById(id);
const {ReviewStore, imagePoint, hitTest, normalizedBox, validateReview, names} = ReviewTools;
const colors = ['#ff85c8', '#ffffff', '#93aaff'];
const store = new ReviewStore();
let source = null, fileName = 'image', originalFileName = '', loadVersion = 0;
let session = null, modelConfig = null, reviewModel = null, imageHash = null, busy = false, loading = false, importing = false, analysed = false;
let zoom = 1, tool = 'pan', gesture = null, draft = null;
const PAGE_SIZE = 8;
let listPage = 0, renderedListKey = '';
const cutoff = () => Number($('confidence').value);
const visible = () => store.visible(cutoff());
const status = message => { $('status').textContent = message; };

function updateControls() {
  const locked = busy || loading || importing || !source;
  for (const id of ['run', 'confidence', 'tool-pan', 'tool-select', 'tool-draw', 'box-class',
    'zoom-out', 'zoom-in', 'zoom-fit', 'zoom-native', 'show-labels']) $(id).disabled = locked;
  $('file').disabled = busy || importing;
  $('load-review').disabled = locked;
  const exportable = !locked && (analysed || store.items.length > 0);
  for (const id of ['save', 'csv', 'json']) $(id).disabled = !exportable;
  $('delete-box').disabled = locked || !store.selected;
  $('undo').disabled = locked || !store.history.length;
  $('restore').disabled = locked || !store.hasCorrections;
  const selected = store.items.find(d => d.uid === store.selected);
  $('selection-info').textContent = selected
    ? `${names[selected.id]} · ${selected.origin === 'manual' ? 'manual box' : `${(selected.score*100).toFixed(0)}% model confidence`} · ${selected.box.map(v => Math.round(v)).join(', ')} px`
    : 'Select a box to inspect or delete it.';
  const items = visible();
  $('total').textContent = items.length;
  for (let id = 0; id < 3; id++) $(['cyto', 'centro', 'cell'][id]).textContent = items.filter(d => d.id === id).length;
  $('manual-count').textContent = items.filter(d => d.origin === 'manual').length;
  $('run').textContent = busy ? 'Analysing…' : 'Run detection';
  $('inference-progress').hidden = !busy;
  const working = busy || loading || importing;
  $('workspace-state').dataset.state = working ? 'busy' : store.hasCorrections ? 'corrected' : 'ready';
  $('workspace-state').textContent = loading ? 'Opening image' : importing ? 'Restoring review' : busy ? 'Analysing image'
    : !source ? 'No image loaded' : store.hasCorrections ? 'Corrections added' : analysed ? 'Ready to review' : 'Image ready';
  renderDetectionList(items,locked);
}

function renderDetectionList(items, locked) {
  const pages = Math.max(1,Math.ceil(items.length/PAGE_SIZE));
  listPage = Math.max(0,Math.min(listPage,pages-1));
  const rows = items.slice(listPage*PAGE_SIZE,(listPage+1)*PAGE_SIZE);
  const key = JSON.stringify([listPage,locked,!!source,analysed,rows.map(d => [d.uid,d.id,d.score,d.origin])]);
  const list = $('detection-list');
  if (key !== renderedListKey) {
    list.replaceChildren();
    if (!rows.length) {
      const empty = document.createElement('p'); empty.className = 'list-empty';
      empty.textContent = !source ? 'Boxes will appear here after detection or manual annotation.'
        : analysed ? 'No visible boxes at this confidence threshold.' : 'Run detection or draw a box to start.';
      list.append(empty);
    }
    for (const d of rows) {
      const button = document.createElement('button');
      button.className = 'detection-row'; button.dataset.uid = d.uid; button.disabled = locked;
      const label = document.createElement('span'); label.className = 'row-name';
      label.textContent = `${names[d.id]} #${d.uid.split('-')[1]}`;
      const score = document.createElement('span'); score.className = 'row-score';
      score.textContent = d.origin === 'manual' ? 'manual' : `${Math.round(d.score*100)}%`;
      button.append(label,score); button.onclick = () => locateDetection(d.uid); list.append(button);
    }
    renderedListKey = key;
  }
  for (const button of list.children) if (button.dataset.uid) button.setAttribute('aria-pressed',String(button.dataset.uid === store.selected));
  $('list-prev').disabled = locked || !items.length || listPage === 0;
  $('list-next').disabled = locked || !items.length || listPage === pages-1;
  $('list-page').textContent = items.length ? `${listPage+1} / ${pages} · ${items.length} boxes` : '0 boxes';
}

function locateDetection(uid) {
  if (!source || busy || loading || importing) return;
  const detection = visible().find(d => d.uid === uid);
  if (!detection) return;
  store.selected = uid; setTool('select');
  const box = detection.box, viewer = $('viewer');
  const closeUp = Math.min(2,(viewer.clientWidth-48)/(box[2]-box[0]),(viewer.clientHeight-48)/(box[3]-box[1]));
  zoomTo(Math.max(zoom,closeUp));
  const bounds = viewer.getBoundingClientRect(), rect = $('canvas').getBoundingClientRect();
  viewer.scrollLeft += rect.left+(box[0]+box[2])/2*zoom-(bounds.left+viewer.clientWidth/2);
  viewer.scrollTop += rect.top+(box[1]+box[3])/2*zoom-(bounds.top+viewer.clientHeight/2);
  status(`Selected ${names[detection.id]} box. Delete selected removes it; Undo restores the last correction.`);
}

$('list-prev').onclick = () => { if (listPage > 0) { listPage--; updateControls(); } };
$('list-next').onclick = () => { if ((listPage+1)*PAGE_SIZE < visible().length) { listPage++; updateControls(); } };

function paint(canvas, exporting = false) {
  if (!source) return;
  if (canvas.width !== source.width || canvas.height !== source.height) {
    canvas.width = source.width; canvas.height = source.height;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(source, 0, 0);
  const unit = exporting ? Math.max(1.5, source.width / 1000) : 2 / zoom;
  ctx.lineWidth = unit;
  ctx.font = `${exporting ? Math.max(14, source.width/120) : 12/zoom}px system-ui`;
  for (const d of visible()) {
    const [x1,y1,x2,y2] = d.box;
    ctx.strokeStyle = colors[d.id];
    ctx.setLineDash(d.origin === 'manual' ? [unit*3, unit*2] : []);
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.setLineDash([]);
    if (!exporting && d.uid === store.selected) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = unit*2;
      ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.lineWidth = unit;
    }
    if ($('show-labels').checked) {
      const text = `${names[d.id]} ${d.origin === 'manual' ? 'manual' : `${(d.score*100).toFixed(0)}%`}`;
      const height = exporting ? Math.max(20, source.width/90) : 18/zoom;
      const width = ctx.measureText(text).width + unit*4;
      const left = Math.max(0, Math.min(x1, canvas.width-width));
      const top = Math.max(0, y1-height);
      ctx.fillStyle = colors[d.id]; ctx.fillRect(left, top, width, height);
      ctx.fillStyle = '#07113c'; ctx.fillText(text, left+unit*2, top+height-unit*2);
    }
  }
  if (!exporting && draft) {
    const box = normalizedBox(draft.start, draft.end, source.width, source.height);
    if (box) {
      ctx.strokeStyle = '#fff'; ctx.setLineDash([unit*3, unit*2]);
      ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]); ctx.setLineDash([]);
    }
  }
}

function render() { if (source) paint($('canvas')); updateControls(); }

function sizeCanvas() {
  if (!source) return;
  const width = source.width*zoom, height = source.height*zoom;
  $('stage').style.width = `${width}px`; $('stage').style.height = `${height}px`;
  $('canvas').style.width = `${width}px`; $('canvas').style.height = `${height}px`;
  $('canvas').style.imageRendering = zoom >= 2 ? 'pixelated' : 'auto';
  $('zoom-value').textContent = `${Math.round(zoom*100)}%`;
}

function zoomTo(value, anchor) {
  if (!source || busy || loading || importing) return;
  const viewer = $('viewer'), bounds = viewer.getBoundingClientRect();
  const at = anchor || [bounds.left+viewer.clientWidth/2, bounds.top+viewer.clientHeight/2];
  const point = imagePoint(at[0], at[1], $('canvas').getBoundingClientRect(), source.width, source.height);
  zoom = Math.max(0.05, Math.min(8, value));
  sizeCanvas();
  const rect = $('canvas').getBoundingClientRect();
  viewer.scrollLeft += rect.left+point[0]*zoom-at[0];
  viewer.scrollTop += rect.top+point[1]*zoom-at[1];
  render();
}

function fit() {
  if (!source) return;
  zoom = Math.max(0.05, Math.min(1, ($('viewer').clientWidth-24)/source.width,
    ($('viewer').clientHeight-24)/source.height));
  sizeCanvas(); $('viewer').scrollLeft = 0; $('viewer').scrollTop = 0; render();
}

function setTool(value) {
  tool = value; cancelGesture(); $('viewer').dataset.tool = tool;
  for (const name of ['pan', 'select', 'draw']) $(`tool-${name}`).setAttribute('aria-pressed', String(name === tool));
  $('tool-help').textContent = {
    pan: 'Pan: drag the image. Use +/− to zoom, or hold Ctrl/⌘ while scrolling.',
    select: 'Select: click inside a box, then Delete selected. Small boxes take priority inside cell boxes.',
    draw: 'Draw box: choose a class, then drag between opposite corners. Escape cancels a drawing.'
  }[tool];
  render();
}

function cancelGesture() {
  if (gesture && $('canvas').hasPointerCapture(gesture.id)) $('canvas').releasePointerCapture(gesture.id);
  gesture = null; draft = null; $('viewer').classList.remove('dragging');
}

$('canvas').addEventListener('pointerdown', event => {
  if (!source || busy || loading || importing || event.button !== 0) return;
  event.preventDefault(); $('viewer').focus({preventScroll:true});
  const point = imagePoint(event.clientX,event.clientY,$('canvas').getBoundingClientRect(),source.width,source.height);
  if (tool === 'select') { store.selected = hitTest(point,visible()); render(); return; }
  gesture = {id:event.pointerId, start:[event.clientX,event.clientY], scroll:[$('viewer').scrollLeft,$('viewer').scrollTop]};
  $('canvas').setPointerCapture(event.pointerId);
  if (tool === 'draw') { draft = {start:point,end:point}; store.selected = null; }
  else $('viewer').classList.add('dragging');
  render();
});
$('canvas').addEventListener('pointermove', event => {
  if (!gesture || gesture.id !== event.pointerId) return;
  if (tool === 'pan') {
    $('viewer').scrollLeft = gesture.scroll[0] - (event.clientX-gesture.start[0]);
    $('viewer').scrollTop = gesture.scroll[1] - (event.clientY-gesture.start[1]);
  } else if (draft) {
    draft.end = imagePoint(event.clientX,event.clientY,$('canvas').getBoundingClientRect(),source.width,source.height);
    paint($('canvas'));
  }
});
$('canvas').addEventListener('pointerup', event => {
  if (!gesture || gesture.id !== event.pointerId) return;
  if (draft) {
    const end = imagePoint(event.clientX,event.clientY,$('canvas').getBoundingClientRect(),source.width,source.height);
    if (Math.hypot(event.clientX-gesture.start[0],event.clientY-gesture.start[1]) >= 3
      && store.add(Number($('box-class').value),draft.start,end,source.width,source.height)) {
      listPage = Math.floor((visible().length-1)/PAGE_SIZE);
      status('Manual box added. Its coordinates are saved in original image pixels.');
    }
  }
  cancelGesture(); render();
});
for (const eventName of ['pointercancel', 'lostpointercapture']) $('canvas').addEventListener(eventName, () => {
  gesture = null; draft = null; $('viewer').classList.remove('dragging'); render();
});
$('viewer').addEventListener('wheel', event => {
  if ((event.ctrlKey || event.metaKey) && source && !busy && !loading && !importing) {
    event.preventDefault(); zoomTo(zoom*Math.exp(-event.deltaY*0.004),[event.clientX,event.clientY]);
  }
}, {passive:false});
$('viewer').addEventListener('keydown', event => {
  if (!source || busy || loading || importing) return;
  if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); removeBox(); }
  else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); undo(); }
  else if (event.key === 'Escape') { cancelGesture(); store.selected = null; render(); }
});
for (const name of ['pan','select','draw']) $(`tool-${name}`).onclick = () => setTool(name);
$('zoom-in').onclick = () => zoomTo(zoom*1.25);
$('zoom-out').onclick = () => zoomTo(zoom/1.25);
$('zoom-native').onclick = () => zoomTo(1);
$('zoom-fit').onclick = fit;
$('show-labels').onchange = render;
function removeBox() { if (store.removeSelected()) { status('Box removed. Undo restores it.'); render(); } }
function undo() { if (store.undo()) { status('Last correction undone.'); render(); } }
$('delete-box').onclick = removeBox;
$('undo').onclick = undo;
$('restore').onclick = () => { if (store.restore()) { listPage = 0; status('Corrections reset to the last model predictions. Undo restores your corrections.'); render(); } };
$('confidence').oninput = event => {
  $('threshold').textContent = Number(event.target.value).toFixed(3);
  listPage = 0;
  if (!visible().some(d => d.uid === store.selected)) store.selected = null;
  if (source) { render(); status('Confidence filter updated. Manual boxes remain visible.'); }
};

$('file').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  if (store.hasCorrections && !window.confirm('Opening another image will replace your corrections. Download your review first if you need it. Continue?')) {
    event.target.value = ''; return;
  }
  const version = ++loadVersion;
  source = null; imageHash = null; reviewModel = null; store.reset(); listPage = 0; analysed = false; loading = true; cancelGesture();
  $('stage').style.display = 'none'; $('empty').style.display = 'block';
  $('image-info').textContent = 'Opening image…'; updateControls(); status('Opening image…');
  try {
    const canvas = document.createElement('canvas'), ctx = canvas.getContext('2d');
    if (/\.tiff?$/i.test(file.name)) {
      if (typeof UTIF === 'undefined') throw Error('TIFF reader could not load. Check your connection.');
      const buffer = await file.arrayBuffer(), ifds = UTIF.decode(buffer);
      if (!ifds.length) throw Error('No TIFF image found.');
      if (ifds.length > 1) throw Error('Only single-page TIFF images are supported.');
      UTIF.decodeImage(buffer,ifds[0]);
      canvas.width = ifds[0].width; canvas.height = ifds[0].height;
      ctx.putImageData(new ImageData(new Uint8ClampedArray(UTIF.toRGBA8(ifds[0])),canvas.width,canvas.height),0,0);
    } else {
      const bitmap = await createImageBitmap(file);
      canvas.width = bitmap.width; canvas.height = bitmap.height;
      ctx.drawImage(bitmap,0,0); bitmap.close();
    }
    if (!canvas.width || !canvas.height) throw Error('Image dimensions are invalid.');
    if (!globalThis.crypto?.subtle) throw Error('Use HTTPS or localhost to enable image fingerprint checks.');
    const digest = await crypto.subtle.digest('SHA-256',await file.arrayBuffer());
    const hash = Array.from(new Uint8Array(digest),byte => byte.toString(16).padStart(2,'0')).join('');
    if (version !== loadVersion) return;
    source = canvas; imageHash = hash; originalFileName = file.name; fileName = file.name.replace(/\.[^.]+$/,'');
    $('canvas').width = canvas.width; $('canvas').height = canvas.height;
    $('stage').style.display = 'block'; $('empty').style.display = 'none';
    $('image-info').textContent = `${file.name} · ${canvas.width} × ${canvas.height} px`;
    loading = false; setTool('pan'); fit();
    status('Image ready. Run detection or draw boxes manually.');
  } catch (error) {
    if (version === loadVersion) { $('image-info').textContent = 'No image selected'; status(`Could not open image: ${error.message}`); }
  } finally { if (version === loadVersion) { loading = false; updateControls(); } }
};

async function loadModel() {
  if (session) return;
  const response = await fetch('model-config.json');
  if (!response.ok) throw Error('Model settings could not load.');
  const config = await response.json();
  if (!Number.isInteger(config.inputSize) || config.inputSize < 32 || config.inputSize > 4096 || config.inputSize%32
    || !Number.isInteger(config.overlap) || config.overlap < 0 || config.overlap >= config.inputSize
    || typeof config.url !== 'string' || !config.url || JSON.stringify(config.classes) !== JSON.stringify(names)) {
    throw Error('Model settings do not match the supported detector.');
  }
  const loaded = await ort.InferenceSession.create(config.url,{executionProviders:['wasm']});
  session = loaded; modelConfig = config;
  $('model-info').textContent = `${config.name} · ${config.inputSize}-pixel inference`;
}

$('run').onclick = async () => {
  if (!source || busy || loading || importing) return;
  if (store.hasCorrections && !window.confirm('Running detection will replace your corrections. Download your review first if you need it. Continue?')) return;
  busy = true; analysed = false; reviewModel = null; cancelGesture(); store.reset(); listPage = 0;
  $('inference-progress').removeAttribute('value'); render();
  try {
    if (typeof ort === 'undefined') throw Error('Inference library could not load. Check your connection.');
    status('Loading model…'); await new Promise(resolve => setTimeout(resolve,30));
    ort.env.wasm.numThreads = 1;
    await loadModel();
    const size = modelConfig.inputSize, width = source.width, height = source.height;
    const whole = modelConfig.mode === 'whole_image';
    const xs = whole ? [0] : tilePositions(width,size,modelConfig.overlap);
    const ys = whole ? [0] : tilePositions(height,size,modelConfig.overlap), all = [];
    const tile = document.createElement('canvas'); tile.width = tile.height = size;
    const ctx = tile.getContext('2d',{willReadFrequently:true});
    let current = 0; const total = xs.length*ys.length;
    for (const y of ys) for (const x of xs) {
      current++;
      status(whole ? 'Analysing whole image…' : `Analysing tile ${current} of ${total}…`); await new Promise(resolve => setTimeout(resolve,0));
      const tw = Math.min(size,width-x), th = Math.min(size,height-y);
      let input, wholeInput;
      if (whole) {
        const rgba = source.getContext('2d').getImageData(0,0,width,height).data;
        wholeInput = wholeImageInput(rgba,width,height,size);
        input = wholeInput.input;
      } else {
        ctx.fillStyle = 'rgb(114,114,114)'; ctx.fillRect(0,0,size,size);
        ctx.drawImage(source,x,y,tw,th,0,0,tw,th);
        const rgba = ctx.getImageData(0,0,size,size).data;
        input = new Float32Array(3*size*size);
        for (let i=0; i<size*size; i++) {
          input[i]=rgba[4*i]/255; input[size*size+i]=rgba[4*i+1]/255; input[2*size*size+i]=rgba[4*i+2]/255;
        }
      }
      const tensor = new ort.Tensor('float32',input,[1,3,size,size]);
      let result;
      try {
        result = await session.run({[session.inputNames[0]]:tensor});
        const out = result[session.outputNames[0]];
        if (out.dims.length !== 3 || out.dims[0] !== 1 || out.dims[2] !== 6) throw Error('Unexpected model output format.');
        for (let i = 0; i < out.data.length; i += 6) {
          const a = out.data, score = a[i+4], id = a[i+5];
          if (!Number.isFinite(score) || score < 0.05 || score > 1 || !Number.isInteger(id) || id < 0 || id > 2) continue;
          const box = whole ? wholeInput.mapBox([a[i],a[i+1],a[i+2],a[i+3]]) : [Math.max(0,Math.min(tw,a[i]))+x,Math.max(0,Math.min(th,a[i+1]))+y,
            Math.max(0,Math.min(tw,a[i+2]))+x,Math.max(0,Math.min(th,a[i+3]))+y];
          if (box.every(Number.isFinite) && box[2]>box[0] && box[3]>box[1]) all.push({id,score,box});
        }
      } finally { tensor.dispose(); if (result) for (const out of Object.values(result)) out.dispose(); }
      $('inference-progress').value = current/total*100;
    }
    store.reset(suppressOverlaps(all)); analysed = true; reviewModel = {...modelConfig,classes:[...names]};
    setTool('select'); status(`Analysis complete: ${visible().length} visible candidate boxes. Select a box to delete it, or draw a missed region.`);
  } catch (error) { status(`Analysis failed: ${error.message}`); }
  finally { busy = false; render(); }
};

function download(blob, name) {
  if (!blob) { status('The export could not be created.'); return; }
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
}
$('save').onclick = () => {
  if (!source || busy) return;
  const canvas = document.createElement('canvas'); paint(canvas,true);
  canvas.toBlob(blob => download(blob,`${fileName}-review.png`));
};
$('csv').onclick = () => download(new Blob([store.csv(cutoff())],{type:'text/csv;charset=utf-8'}),`${fileName}-detections.csv`);
$('json').onclick = () => {
  const review = {schema_version:2, file:originalFileName, source_file_sha256:imageHash, image_width:source.width,image_height:source.height,
    confidence_threshold:cutoff(), coordinate_system:'original image pixels, xyxy',
    model:analysed ? reviewModel : null, ...store.snapshot(cutoff())};
  download(new Blob([JSON.stringify(review,null,2)],{type:'application/json'}),`${fileName}-review.json`);
};
$('load-review').onclick = () => { $('review-file').value = ''; $('review-file').click(); };
$('review-file').onchange = async event => {
  const file = event.target.files[0];
  if (!file || !source || busy || loading || importing) return;
  importing = true; cancelGesture(); updateControls();
  try {
    if (file.size > 10*1024*1024) throw Error('Review files must be smaller than 10 MB.');
    const review = JSON.parse(await file.text());
    const validated = validateReview(review,source.width,source.height,imageHash);
    if ((store.items.length || store.original.length) && !window.confirm('Loading this saved review replaces the current boxes and starts a new undo history. Continue?')) return;
    store.loadReview(validated);
    listPage = 0;
    reviewModel = validated.model; analysed = reviewModel !== null;
    $('confidence').value = String(validated.cutoff);
    $('threshold').textContent = validated.cutoff.toFixed(3);
    setTool('select');
    status('Saved review restored. The original image fingerprint matches.');
  } catch (error) { status(`Could not load review: ${error.message}`); }
  finally { importing = false; event.target.value = ''; render(); }
};
updateControls();
