'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const root = path.resolve(__dirname,'..');
const dist = path.join(root,'dist');
const config = JSON.parse(fs.readFileSync(path.join(dist,'model-config.json'),'utf8'));
const manifest = JSON.parse(fs.readFileSync(path.join(root,'deployment-model.json'),'utf8'));
assert.deepEqual(manifest.input,[1,3,1024,1024],'Verified model input must be 1024 square.');
assert.deepEqual(config.classes,manifest.classes,'Deployment class IDs differ from the verified model.');
if (config.url !== manifest.file || config.inputSize !== 1024 || config.mode !== 'whole_image') {
  throw Error('Deployment model configuration differs from the verified model.');
}
const model = fs.readFileSync(path.join(dist,config.url));
if (model.length !== manifest.bytes || crypto.createHash('sha256').update(model).digest('hex') !== manifest.sha256) {
  throw Error('The selected ONNX model is missing or its bytes have changed.');
}
const html = fs.readFileSync(path.join(dist,'index.html'),'utf8');
const confidenceInput = html.match(/<input\b[^>]*\bid="confidence"[^>]*>/)?.[0];
const initialConfidence = confidenceInput?.match(/\bvalue="([^"]+)"/)?.[1];
assert.equal(Number(initialConfidence),manifest.confidence,'Initial confidence differs from the evaluated cutoff.');
const {suppressOverlaps} = require(path.join(dist,'tiling.js'));
const reference = {id:0,score:0.9,box:[0,0,10,10]};
// Bracket the cutoff to catch changes to the app\'s default suppression threshold.
for (const iou of [manifest.overlap_suppression_iou-0.001,manifest.overlap_suppression_iou+0.001]) {
  const shift = 10*(1-iou)/(1+iou);
  const candidate = {id:0,score:0.8,box:[shift,0,10+shift,10]};
  assert.equal(suppressOverlaps([reference,candidate]).length,iou > manifest.overlap_suppression_iou ? 1 : 2,
    'Default overlap suppression differs from the evaluated cutoff.');
}
for (const [,src] of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
  if (!/^(?:https?:|#|data:)/.test(src) && !fs.existsSync(path.join(dist,src))) {
    throw Error('Missing site asset: '+src);
  }
}
for (const name of ['app.js','review.js','tiling.js']) {
  execFileSync(process.execPath,['--check',path.join(dist,name)],{stdio:'inherit'});
}
console.log('Selected ONNX hash, class mapping, evaluated cutoffs and static assets verified.');
