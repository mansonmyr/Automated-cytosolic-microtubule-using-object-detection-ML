const test = require('node:test');
const assert = require('node:assert/strict');
const {ReviewStore, imagePoint, normalizedBox, hitTest} = require('../dist/review.js');
const {tilePositions, suppressOverlaps} = require('../dist/tiling.js');

test('a display rectangle maps back to native image pixels after zoom or pan', () => {
  assert.deepEqual(imagePoint(350,200,{left:100,top:100,width:1024,height:1024},2048,2048),[500,200]);
  assert.deepEqual(imagePoint(150,200,{left:-350,top:0,width:2048,height:2048},2048,2048),[500,200]);
  assert.deepEqual(imagePoint(-1,9999,{left:0,top:0,width:512,height:512},2048,2048),[0,2048]);
});
test('dragging in either direction clips boxes and rejects accidental clicks', () => {
  assert.deepEqual(normalizedBox([300,400],[-20,100],2048,2048),[0,100,300,400]);
  assert.equal(normalizedBox([10,10],[10.5,11],2048,2048),null);
  assert.equal(normalizedBox([NaN,10],[20,30],2048,2048),null);
});
test('a small object can be selected inside a large cell box', () => {
  assert.equal(hitTest([110,110],[{uid:'cell',box:[0,0,500,500]},{uid:'small',box:[100,100,120,120]}]),'small');
});
test('corrections survive filtering and export never invents a manual confidence', () => {
  const store = new ReviewStore();
  store.reset([{id:0,score:0.8,box:[10,20,30,40]},{id:2,score:0.2,box:[0,0,100,100]}]);
  assert.equal(store.add(1,[50,60],[80,90],2048,2048),true);
  assert.equal(store.visible(0.95).length,1);
  assert.equal(store.visible(0.05).length,3);
  assert.match(store.csv(0.95),/manual-1,centrosomal,,50.00,60.00,80.00,90.00,manual/);
  store.selected = 'model-1'; store.removeSelected();
  const audit = store.snapshot(0.25);
  assert.equal(audit.removed_predictions[0].uid,'model-1');
  assert.equal(audit.original_predictions.length,2);
  assert.equal(store.undo(),true);
  assert.equal(store.items.length,3);
  assert.equal(store.restore(),true);
  assert.equal(store.items.length,2);
  assert.equal(store.undo(),true);
  assert.equal(store.items.length,3);
});
test('manual IDs remain unique after undo and a new drawing', () => {
  const store = new ReviewStore();
  store.add(0,[1,1],[10,10],100,100); store.undo(); store.add(0,[20,20],[40,40],100,100);
  assert.equal(store.items[0].uid,'manual-2');
  store.reset(); assert.equal(store.history.length,0); assert.equal(store.hasCorrections,false);
});
test('both supported input sizes cover the original image without resizing', () => {
  assert.deepEqual(tilePositions(2048,2048,0),[0]);
  assert.deepEqual(tilePositions(2048,640,160),[0,480,960,1408]);
  assert.throws(() => tilePositions(2048,640,640));
  const boxes=[{id:0,score:0.9,box:[0,0,10,10]},{id:0,score:0.7,box:[0,0,10,10]},{id:1,score:0.8,box:[0,0,10,10]}];
  assert.equal(suppressOverlaps(boxes).length,2);
});
