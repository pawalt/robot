/* State-only radial policy and matching continuous Acrobot dynamics. */
(function(root){
'use strict';
const PI=Math.PI,TAU=2*PI;
const wrap=x=>((x+PI)%TAU+TAU)%TAU-PI;
const clip=(x,a,b)=>Math.max(a,Math.min(b,x));
function policy(model,s){
 const z=[Math.sin(s[0]),Math.cos(s[0]),Math.sin(s[1]),Math.cos(s[1]),s[2]/4,s[3]/8];
 const n=model.centers.length,logits=new Float64Array(n),values=new Float64Array(n);let max=-Infinity;
 for(let j=0;j<n;j++){
  let d=0,u=model.bias[j];for(let k=0;k<6;k++){const e=z[k]-model.centers[j][k];d+=e*e;u+=z[k]*model.weights[j][k]}
  logits[j]=-d/(2*model.sigma*model.sigma);values[j]=u;max=Math.max(max,logits[j]);
 }
 let sum=0,total=0;for(let j=0;j<n;j++){const w=Math.exp(logits[j]-max);sum+=w;total+=w*values[j]}
 return clip(total/sum,-1,1);
}
function derivative(s,u){
 const [a,b,v,w]=s,c=Math.cos(b),sn=Math.sin(b),d1=3.5+c,d2=1.25+.5*c;
 const p2=4.9*Math.cos(a+b-PI/2);
 const p1=-.5*w*w*sn-w*v*sn+14.7*Math.cos(a-PI/2)+p2;
 const a2=(u+d2/d1*p1-.5*v*v*sn-p2)/(1.25-d2*d2/d1);
 return [v,w,-(d2*a2+p1)/d1,a2];
}
function physics(s,u,h=.0025){
 const add=(a,b,k)=>a.map((v,i)=>v+k*b[i]);
 const k1=derivative(s,u),k2=derivative(add(s,k1,h/2),u),k3=derivative(add(s,k2,h/2),u),k4=derivative(add(s,k3,h),u);
 const x=s.map((v,i)=>v+h/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]));
 return [wrap(x[0]),wrap(x[1]),clip(x[2],-4*PI,4*PI),clip(x[3],-9*PI,9*PI)];
}
function balanced(s){return -Math.cos(s[0])>Math.cos(PI/18)&&-Math.cos(s[0]+s[1])>Math.cos(PI/18)&&Math.abs(s[2])<.5&&Math.abs(s[2]+s[3])<.5}
function step(model,s,enabled=true){const u=enabled?policy(model,s):0;let x=s,hold=0;for(let i=0;i<20;i++){x=physics(x,u);hold=balanced(x)?hold+.0025:0}return {state:x,torque:u,holdTail:hold,allBalanced:hold>.049999}}
const api={policy,derivative,physics,balanced,step,wrap};
if(typeof module!=='undefined'&&module.exports)module.exports=api;root.Acrobot=api;
})(typeof window==='undefined'?globalThis:window);
