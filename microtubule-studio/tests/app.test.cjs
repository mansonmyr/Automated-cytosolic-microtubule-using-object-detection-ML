const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function app() {
  const downloads = [], elements = {}, blobs = new Map();
  const html = fs.readFileSync(path.join(__dirname,'../dist/index.html'),'utf8');
  const ctx = {clearRect(){},drawImage(){},setLineDash(){},strokeRect(){},fillRect(){},fillText(){},
    measureText(text){return {width:text.length*7}},getImageData(x,y,w,h){return {data:new Uint8ClampedArray(w*h*4)}}};
  function element(id) {
    const handlers = {}, capture = new Set();
    return {id,style:{},dataset:{},textContent:'',value:'',checked:false,disabled:true,width:300,height:150,
      clientWidth:824,clientHeight:600,scrollLeft:0,scrollTop:0,classList:{add(){},remove(){}},
      getContext:()=>ctx,setAttribute(){},focus(){},addEventListener(name,fn){handlers[name]=fn},
      fire(name,event){handlers[name]?.(event)},setPointerCapture(id){capture.add(id)},
      hasPointerCapture:id=>capture.has(id),releasePointerCapture(id){capture.delete(id)},
      getBoundingClientRect(){return {left:400-elements.viewer.scrollLeft,top:200-elements.viewer.scrollTop,
        width:parseFloat(this.style.width)||824,height:parseFloat(this.style.height)||600}},
      click(){downloads.push({name:this.download,blob:blobs.get(this.href)})},
      toBlob(fn){fn(new Blob(['image']))}};
  }
  for (const [,id] of html.matchAll(/id="([^"]+)"/g)) elements[id] = element(id);
  elements.confidence.value = '0.25'; elements['show-labels'].checked = true;
  const config = {name:'Test native model',url:'test.onnx',inputSize:2048,overlap:0,classes:['cytosolic','centrosomal','cell']};
  const state = {runs:0, tensorShapes:[], confirm:true};
  const ort = {env:{wasm:{}},Tensor:class {constructor(type,input,dims){state.tensorShapes.push(dims)} dispose(){}},
    InferenceSession:{async create(){return {inputNames:['images'],outputNames:['output'],async run(){
      state.runs++; return {output:{dims:[1,2,6],data:new Float32Array([100,100,130,140,.8,0,300,300,600,600,.2,2]),dispose(){}}};
    }}}}};
  const sandbox = {console,Blob,Float32Array,Uint8ClampedArray,setTimeout,clearTimeout,ort,
    window:{confirm:()=>state.confirm},document:{getElementById(id){assert.ok(elements[id],`Missing element ${id}`);return elements[id]},createElement:tag=>element(tag)},
    URL:{createObjectURL(blob){const id=String(blobs.size);blobs.set(id,blob);return id},revokeObjectURL(){}},
    fetch:async()=>({ok:true,json:async()=>config}),
    createImageBitmap:async file=>{if(file.fail)throw Error('Invalid image');return {width:2048,height:2048,close(){}}}};
  vm.createContext(sandbox);
  for (const file of ['tiling.js','review.js','app.js']) vm.runInContext(fs.readFileSync(path.join(__dirname,'../dist',file),'utf8'),sandbox);
  return {elements,state,sandbox,downloads,
    open:file=>elements.file.onchange({target:{files:[file],value:file.name}}),
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
