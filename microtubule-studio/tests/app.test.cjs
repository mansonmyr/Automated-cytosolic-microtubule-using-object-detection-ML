const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function app(modelOverride = null) {
  const downloads = [], elements = {}, blobs = new Map();
  const html = fs.readFileSync(path.join(__dirname,'../dist/index.html'),'utf8');
  const ctx = {clearRect(){},drawImage(){},setLineDash(){},strokeRect(){},fillRect(){},fillText(){},
    measureText(text){return {width:text.length*7}},getImageData(x,y,w,h){return {data:new Uint8ClampedArray(w*h*4)}}};
  function element(id) {
    const handlers = {}, capture = new Set();
    return {id,style:{},dataset:{},children:[],textContent:'',value:'',checked:false,disabled:true,width:300,height:150,
      clientWidth:824,clientHeight:600,scrollLeft:0,scrollTop:0,classList:{add(){},remove(){}},
      getContext:()=>ctx,setAttribute(name,value){this[name]=value},removeAttribute(name){delete this[name]},
      append(...nodes){this.children.push(...nodes)},replaceChildren(...nodes){this.children=[...nodes]},
      focus(){},addEventListener(name,fn){handlers[name]=fn},
      fire(name,event){handlers[name]?.(event)},setPointerCapture(id){capture.add(id)},
      hasPointerCapture:id=>capture.has(id),releasePointerCapture(id){capture.delete(id)},
      getBoundingClientRect(){if(id==='viewer')return {left:380,top:180,width:824,height:600};return {left:400-elements.viewer.scrollLeft,top:200-elements.viewer.scrollTop,
        width:parseFloat(this.style.width)||824,height:parseFloat(this.style.height)||600}},
      click(){downloads.push({name:this.download,blob:blobs.get(this.href)})},
      toBlob(fn){fn(new Blob(['image']))}};
  }
  for (const [,id] of html.matchAll(/id="([^"]+)"/g)) elements[id] = element(id);
  elements.confidence.value = '0.25'; elements['show-labels'].checked = true;
  const config = modelOverride ?? {name:'Test native model',url:'test.onnx',inputSize:2048,overlap:0,classes:['cytosolic','centrosomal','cell']};
  const state = {runs:0, tensorShapes:[], confirm:true};
  const ort = {env:{wasm:{}},Tensor:class {constructor(type,input,dims){state.tensorShapes.push(dims)} dispose(){}},
    InferenceSession:{async create(){return {inputNames:['images'],outputNames:['output'],async run(){
      state.runs++; return {output:{dims:[1,2,6],data:new Float32Array([100,100,130,140,.8,0,300,300,600,600,.2,2]),dispose(){}}};
    }}}}};
  const sandbox = {console,Blob,Float32Array,Uint8ClampedArray,setTimeout,clearTimeout,ort,crypto:require('node:crypto').webcrypto,
    window:{confirm:()=>state.confirm},document:{getElementById(id){assert.ok(elements[id],`Missing element ${id}`);return elements[id]},createElement:tag=>element(tag)},
    URL:{createObjectURL(blob){const id=String(blobs.size);blobs.set(id,blob);return id},revokeObjectURL(){}},
    fetch:async()=>({ok:true,json:async()=>config}),
    createImageBitmap:async file=>{if(file.fail)throw Error('Invalid image');return {width:2048,height:2048,close(){}}}};
  vm.createContext(sandbox);
  for (const file of ['tiling.js','review.js','app.js']) vm.runInContext(fs.readFileSync(path.join(__dirname,'../dist',file),'utf8'),sandbox);
  return {elements,state,sandbox,downloads,
    open:file=>elements.file.onchange({target:{files:[{...file,arrayBuffer:async()=>new TextEncoder().encode(file.bytes??file.name).buffer}],value:file.name}}),
    importReview:review=>elements['review-file'].onchange({target:{files:[{size:1000,text:async()=>JSON.stringify(review)}],value:'review.json'}}),
    draw(a,b){const rect=elements.canvas.getBoundingClientRect();const p=point=>({clientX:rect.left+point[0]*rect.width/2048,clientY:rect.top+point[1]*rect.height/2048,
      pointerId:1,button:0,preventDefault(){}});elements['tool-draw'].onclick();elements.canvas.fire('pointerdown',p(a));elements.canvas.fire('pointerup',p(b));}};
}

test('native model input, drawing, filtering and exports retain image coordinates', async () => {
  const {elements:e,state,open,draw,downloads} = app();
  await open({name:'sample.png'});
  assert.equal(e.run.disabled,false);
  await e.run.onclick();
  assert.equal(state.runs,1);
  assert.deepEqual(Array.from(state.tensorShapes[0]),[1,3,2048,2048]);
  assert.equal(e.total.textContent,1);
  e['zoom-native'].onclick(); e['box-class'].value='1'; draw([800,900],[700,750]);
  assert.equal(e.total.textContent,2);
  e.confidence.value='0.95';e.confidence.oninput({target:e.confidence});
  assert.equal(e.total.textContent,1);assert.equal(state.runs,1);
  e.csv.onclick();const csv=await downloads.at(-1).blob.text();
  assert.match(csv,/centrosomal,,700.00,750.00,800.00,900.00,manual/);
  e.json.onclick();const json=JSON.parse(await downloads.at(-1).blob.text());
  assert.equal(json.image_width,2048);assert.equal(json.model.inputSize,2048);assert.equal(json.detections[0].score,null);
  e['delete-box'].onclick();assert.equal(e.total.textContent,0);
  e.undo.onclick();assert.equal(e.total.textContent,1);
  e.restore.onclick();assert.equal(e.total.textContent,0);
  e.undo.onclick();assert.equal(e.total.textContent,1);
  e.save.onclick();assert.equal(downloads.at(-1).name,'sample-review.png');
});
test('failed image loads and failed reruns clear old results', async () => {
  const {elements:e,sandbox,open}=app();
  await open({name:'first.png'});await e.run.onclick();assert.equal(e.total.textContent,1);
  sandbox.ort=undefined;await e.run.onclick();assert.equal(e.total.textContent,0);assert.equal(e.csv.disabled,true);
  await open({name:'bad.png',fail:true});assert.equal(e.total.textContent,0);assert.equal(e.run.disabled,true);
  assert.equal(e.stage.style.display,'none');assert.match(e.status.textContent,/Could not open image/);
});
test('an older asynchronous upload cannot replace the latest image', async () => {
  const {elements:e,sandbox,open}=app();
  let resolveOld;
  sandbox.createImageBitmap=file=>file.name==='old.png'?new Promise(resolve=>{resolveOld=resolve}):Promise.resolve({width:2048,height:2048,close(){}});
  const old=open({name:'old.png'});await open({name:'new.png'});
  resolveOld({width:100,height:100,close(){}});await old;
  assert.match(e['image-info'].textContent,/new.png/);assert.equal(e.canvas.width,2048);
});

test('a downloaded review reopens on the matching image and remains editable', async () => {
  const original=app();await original.open({name:'sample.png'});await original.elements.run.onclick();
  original.elements['box-class'].value='1';original.draw([700,750],[800,900]);
  original.elements.json.onclick();const saved=JSON.parse(await original.downloads.at(-1).blob.text());
  assert.equal(saved.schema_version,2);assert.match(saved.source_file_sha256,/^[a-f0-9]{64}$/);
  const resumed=app();await resumed.open({name:'renamed.png',bytes:'sample.png'});
  await resumed.importReview(saved);
  assert.equal(resumed.elements.total.textContent,2);assert.equal(resumed.state.runs,0);
  assert.match(resumed.elements.status.textContent,/fingerprint matches/);
  resumed.elements['box-class'].value='0';resumed.draw([1500,1500],[1600,1600]);
  resumed.elements.json.onclick();const next=JSON.parse(await resumed.downloads.at(-1).blob.text());
  assert.equal(next.all_detections.filter(d=>d.origin==='manual').length,2);
  assert.equal(new Set(next.all_detections.map(d=>d.uid)).size,next.all_detections.length);
});
test('a bad import preserves the current corrections', async () => {
  const a=app();await a.open({name:'sample.png'});a.elements['box-class'].value='1';a.draw([100,100],[200,200]);
  a.elements.json.onclick();const saved=JSON.parse(await a.downloads.at(-1).blob.text());
  saved.source_file_sha256='f'.repeat(64);await a.importReview(saved);
  assert.equal(a.elements.total.textContent,1);assert.match(a.elements.status.textContent,/different image file/);
  assert.equal(a.elements['load-review'].disabled,false);
});
test('imported model provenance cannot change the active inference session', async () => {
  const a=app();await a.open({name:'sample.png'});await a.elements.run.onclick();a.elements.json.onclick();
  const saved=JSON.parse(await a.downloads.at(-1).blob.text());saved.model.inputSize=640;saved.model.overlap=160;
  await a.importReview(saved);await a.elements.run.onclick();
  assert.deepEqual(Array.from(a.state.tensorShapes.at(-1)),[1,3,2048,2048]);
});

test('the box list locates a detection and deletion updates the list and counts', async () => {
  const a=app();await a.open({name:'sample.png'});await a.elements.run.onclick();
  const row=a.elements['detection-list'].children[0];
  assert.equal(row.dataset.uid,'model-1');row.onclick();
  assert.equal(row['aria-pressed'],'true');assert.equal(a.elements['zoom-value'].textContent,'200%');
  assert.match(a.elements['selection-info'].textContent,/cytosolic/);
  a.elements['delete-box'].onclick();assert.equal(a.elements.total.textContent,0);
  assert.equal(a.elements['list-page'].textContent,'0 boxes');
  a.elements.undo.onclick();assert.equal(a.elements.total.textContent,1);
});
test('box pagination stays within the available pages after corrections', async () => {
  const a=app();await a.open({name:'sample.png'});a.elements['box-class'].value='0';
  for(let i=0;i<9;i++)a.draw([100+i*30,100],[120+i*30,120]);
  assert.equal(a.elements['list-page'].textContent,'2 / 2 · 9 boxes');
  assert.equal(a.elements['detection-list'].children.length,1);
  a.elements['list-prev'].onclick();assert.equal(a.elements['detection-list'].children.length,8);
  a.elements['detection-list'].children[0].onclick();a.elements['delete-box'].onclick();
  assert.equal(a.elements['list-page'].textContent,'1 / 1 · 8 boxes');
  assert.equal(a.elements['list-next'].disabled,true);
  a.elements.undo.onclick();assert.equal(a.elements['list-next'].disabled,false);
});
test('workspace status and progress reflect actual inference activity', async () => {
  const a=app();assert.equal(a.elements['workspace-state'].textContent,'No image loaded');
  await a.open({name:'sample.png'});assert.equal(a.elements['workspace-state'].textContent,'Image ready');
  const running=a.elements.run.onclick();
  assert.equal(a.elements['workspace-state'].textContent,'Analysing image');
  assert.equal(a.elements['inference-progress'].hidden,false);
  await running;
  assert.equal(a.elements['workspace-state'].textContent,'Ready to review');
  assert.equal(a.elements['inference-progress'].hidden,true);
  assert.equal(a.elements['inference-progress'].value,100);
});

test('selected whole-image detector uses one pass and restores native coordinates', async () => {
  const config = JSON.parse(fs.readFileSync(path.join(__dirname,'../dist/model-config.json'),'utf8'));
  const {elements:e,open,state,downloads}=app(config);
  await open({name:'square.png'});
  e.confidence.value='0.174';
  await e.run.onclick();
  assert.deepEqual(Array.from(state.tensorShapes[0]),[1,3,1024,1024]);
  assert.equal(state.runs,1);
  e.json.onclick();
  const saved=JSON.parse(await downloads.at(-1).blob.text());
  assert.equal(saved.confidence_threshold,0.174);
  assert.equal(saved.model.mode,'whole_image');
  assert.deepEqual(saved.original_predictions[0].box,[200,200,260,280]);
});
