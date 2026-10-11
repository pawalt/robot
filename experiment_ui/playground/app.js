'use strict';
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d'),C=window.Acrobot;
let model=null,state=[Math.PI+Math.PI/12,0,0,0],pose=state.slice(),running=false,time=0,hold=0,torque=0,accumulator=0,last=null,drag=null,trail=[],eventUntil=0;
let width=800,height=520,scale=120,pivot=[400,260];
function joints(){const elbow=[pivot[0]+scale*Math.sin(state[0]),pivot[1]+scale*Math.cos(state[0])];return [pivot,elbow,[elbow[0]+scale*Math.sin(state[0]+state[1]),elbow[1]+scale*Math.cos(state[0]+state[1])]]}
function resize(){const r=canvas.getBoundingClientRect();width=r.width;height=r.height;const dpr=window.devicePixelRatio||1;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);scale=Math.min((height-70)/4,(width-60)/4);pivot=[width/2,height/2];draw()}
function sliders(){const upper=Math.round(C.wrap(state[0]-Math.PI)*180/Math.PI),elbow=Math.round(C.wrap(state[1])*180/Math.PI);$('upper').value=upper;$('elbow').value=elbow;$('upperValue').textContent=upper+'°';$('elbowValue').textContent=elbow+'°'}
function controls(){ $('run').textContent=running?'Pause':(time>0?'Resume':'Release');$('hold').textContent=hold.toFixed(1)+' s';$('time').textContent=time.toFixed(1);const magnitude=Math.abs(torque),positive=torque>=0;$('effortValue').textContent=(magnitude<.005?'0.00':(positive?'+':'')+torque.toFixed(2))+' N·m';$('effort').style.width=magnitude*50+'%';$('effort').style.left=(positive?50:50-magnitude*50)+'%';$('effort').style.background=positive?'#81e5c9':'#edc780';const text=!model?'Loading policy…':!running?(time>0?'Paused':'Ready to release'):!$('motor').checked?'Motor off':hold>=10?'Balanced':hold>0?'Settling':'Finding balance';if($('status').textContent!==text)$('status').textContent=text; }
function setPose(s){eventUntil=0;$('event').style.display='none';running=false;state=s.slice();state[2]=state[3]=0;pose=state.slice();time=hold=torque=accumulator=0;trail=[];sliders();controls();draw()}
function event(text){$('event').textContent=text;eventUntil=performance.now()+2200;$('event').style.display='block'}
function draw(){ctx.clearRect(0,0,width,height);ctx.strokeStyle='#1c313c';ctx.lineWidth=1;for(let x=width%32;x<width;x+=32){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,height);ctx.stroke()}for(let y=height%32;y<height;y+=32){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(width,y);ctx.stroke()}
ctx.setLineDash([4,6]);ctx.strokeStyle='#466069';ctx.beginPath();ctx.moveTo(pivot[0],25);ctx.lineTo(pivot[0],height-25);ctx.stroke();ctx.setLineDash([]);
ctx.strokeStyle='#edc780';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pivot[0]-36,pivot[1]-2*scale);ctx.lineTo(pivot[0]+36,pivot[1]-2*scale);ctx.stroke();ctx.fillStyle='#9eb6c1';ctx.font='12px system-ui';ctx.fillText('UPRIGHT',pivot[0]+45,pivot[1]-2*scale+4);
if(trail.length>1){ctx.beginPath();for(let i=0;i<trail.length;i++){const p=trail[i];const x=pivot[0]+scale*p[0],y=pivot[1]+scale*p[1];if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}ctx.strokeStyle='#81e5c92b';ctx.lineWidth=2;ctx.stroke()}
const p=joints();ctx.lineCap='round';for(let i=0;i<2;i++){ctx.beginPath();ctx.moveTo(...p[i]);ctx.lineTo(...p[i+1]);ctx.lineWidth=12;ctx.strokeStyle=i?'#b8d5e2':'#81e5c9';ctx.stroke()}
for(let i=0;i<3;i++){ctx.beginPath();ctx.arc(...p[i],i===0?8:11,0,Math.PI*2);ctx.fillStyle='#edc780';ctx.fill();if(i){ctx.beginPath();ctx.arc(...p[i],18,0,Math.PI*2);ctx.strokeStyle='#edc78055';ctx.lineWidth=1;ctx.stroke()}}
drawMotor(p[1]);
if(!running){ctx.fillStyle='#9eb6c1';ctx.textAlign='center';ctx.fillText('DRAG THE JOINTS TO SET YOUR DROP',width/2,height-12);ctx.textAlign='left'}}
// Positive joint torque increases theta2: counterclockwise in our x=sin, y=cos view.
function drawMotor(elbow){
 const [x,y]=elbow,magnitude=Math.abs(torque),color=torque>=0?'#81e5c9':'#edc780';
 ctx.beginPath();ctx.arc(x,y,15,0,2*Math.PI);ctx.fillStyle='#122832';ctx.fill();ctx.strokeStyle='#edc780';ctx.lineWidth=3;ctx.stroke();
 ctx.fillStyle='#edc780';ctx.font='bold 11px system-ui';ctx.textAlign='center';ctx.fillText('M',x,y+4);
 const labelX=Math.max(50,Math.min(width-50,x)),labelY=y+53;
 if(!running&&time===0){ctx.fillStyle='#0b141be8';ctx.fillRect(labelX-43,labelY-13,86,20);ctx.fillStyle='#edc780';ctx.font='10px system-ui';ctx.fillText('ELBOW MOTOR',labelX,labelY);}
 if(magnitude>=.005){
  const direction=torque>0?-1:1,radius=29,start=-Math.PI/2,sweep=.35+magnitude*1.65*Math.PI,end=start+direction*sweep;
  ctx.beginPath();ctx.arc(x,y,radius,start,end,direction<0);ctx.strokeStyle=color;ctx.lineWidth=2+3*magnitude;ctx.stroke();
  const tx=x+radius*Math.cos(end),ty=y+radius*Math.sin(end),heading=end+direction*Math.PI/2,size=7+3*magnitude;
  ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-size*Math.cos(heading-.5),ty-size*Math.sin(heading-.5));ctx.lineTo(tx-size*Math.cos(heading+.5),ty-size*Math.sin(heading+.5));ctx.closePath();ctx.fillStyle=color;ctx.fill();
 }
 ctx.textAlign='left';
}
function animate(now){const elapsed=last===null?0:Math.min((now-last)/1000,.1);last=now;if(running&&model){accumulator+=elapsed*Number($('speed').value);while(accumulator>=.05){const r=C.step(model,state,$('motor').checked);state=r.state;torque=r.torque;hold=r.allBalanced?hold+.05:r.holdTail;time+=.05;accumulator-=.05;if(!state.every(Number.isFinite)){running=false;event('Simulation stopped. Reset to try again.');break}trail.push([Math.sin(state[0])+Math.sin(state[0]+state[1]),Math.cos(state[0])+Math.cos(state[0]+state[1])]);if(trail.length>150)trail.shift()}controls();draw()}if(now>eventUntil)$('event').style.display='none';requestAnimationFrame(animate)}
$('run').onclick=()=>{if(!model)return;running=!running;accumulator=0;controls();draw()};$('reset').onclick=()=>setPose(pose);
$('motor').onchange=()=>{torque=0;event($('motor').checked?'Neural control on':'Motor off — free fall');controls();draw()};
$('kick').onclick=()=>{if(!model)return;state[2]=Math.min(4*Math.PI,state[2]+4);state[3]=Math.max(-9*Math.PI,state[3]-4);hold=0;running=true;event('KICK · velocity changed');controls()};
for(const id of ['upper','elbow'])$(id).oninput=()=>setPose([Math.PI+Number($('upper').value)*Math.PI/180,Number($('elbow').value)*Math.PI/180,0,0]);
for(const b of document.querySelectorAll('[data-preset]'))b.onclick=()=>setPose(({tilt:[Math.PI+Math.PI/12,0,0,0],top:[Math.PI,0,0,0],side:[Math.PI/2,0,0,0],bottom:[0,0,0,0]})[b.dataset.preset]);
canvas.onpointerdown=e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,p=joints();drag=Math.hypot(x-p[1][0],y-p[1][1])<Math.hypot(x-p[2][0],y-p[2][1])?1:2;canvas.setPointerCapture(e.pointerId);setPose(state)};
canvas.onpointermove=e=>{if(!drag)return;const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,p=joints(),origin=drag===1?p[0]:p[1];const angle=Math.atan2(x-origin[0],y-origin[1]);const next=state.slice();if(drag===1)next[0]=angle;else next[1]=C.wrap(angle-state[0]);setPose(next)};
canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;pose=state.slice();draw()};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running){running=false;accumulator=0;controls()}last=null});
new ResizeObserver(resize).observe(canvas);sliders();controls();requestAnimationFrame(animate);
fetch('/playground/policy.json?v=1').then(r=>{if(!r.ok)throw Error('Policy download failed');return r.json()}).then(p=>{if(p.centers?.length!==467||p.weights?.length!==467||p.bias?.length!==467||!(p.sigma>0))throw Error('Invalid policy file');model=p;$('run').disabled=$('kick').disabled=false;controls()}).catch(e=>{$('status').textContent='Could not load policy. Reload to retry.';console.error(e)});
