(function(root){
'use strict';
const C=typeof module!=='undefined'?require('./cells.js'):root.CellWorkflow;
function describe(points,rgba,width,height){
 const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
 const x0=Math.max(0,Math.floor(Math.min(...xs))),x1=Math.min(width,Math.ceil(Math.max(...xs)));
 const y0=Math.max(0,Math.floor(Math.min(...ys))),y1=Math.min(height,Math.ceil(Math.max(...ys)));
 const area=C.area(points),values=[];let perimeter=0;
 points.forEach((p,i)=>{const q=points[(i+1)%points.length];perimeter+=Math.hypot(p[0]-q[0],p[1]-q[1]);});
 const step=Math.max(2,Math.ceil(Math.max(x1-x0,y1-y0)/100));
 for(let y=y0;y<y1;y+=step)for(let x=x0;x<x1;x+=step)if(C.inside([x+.5,y+.5],points))values.push(rgba[(y*width+x)*4+1]/255);
 values.sort((a,b)=>a-b);const n=Math.max(values.length,1);
 return [Math.log(Math.max(area/(width*height),1e-8)),Math.log(Math.max((x1-x0)/Math.max(y1-y0,1),1e-8)),area/Math.max((x1-x0)*(y1-y0),1),4*Math.PI*area/Math.max(perimeter*perimeter,1),values.reduce((a,b)=>a+b,0)/n,values[Math.floor((n-1)*.9)]??0];
}
const tolerances=[.2,.2,.08,.08,.04,.08];
const distance=(a,b)=>Math.max(...a.map((x,i)=>Math.abs(x-b[i])/tolerances[i]));
function classify(features,examples){
 const negative=Math.min(...examples.excluded.map(x=>distance(features,x.features)));
 const positive=Math.min(...examples.retained.map(x=>distance(features,x.features)));
 return {exclude:negative<=1 && negative<positive*.7,negativeDistance:negative,positiveDistance:positive};
}
const api={describe,classify};if(typeof module!=='undefined')module.exports=api;root.CellEligibility=api;
})(typeof globalThis!=='undefined'?globalThis:this);
