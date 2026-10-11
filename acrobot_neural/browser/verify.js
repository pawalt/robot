const fs=require('fs'),C=require('./core.js');
const model=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const samples=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const torques=samples.states.map(s=>C.policy(model,s));
const transitions=samples.states.map((s,i)=>C.physics(s,samples.torques[i]));
const cases=[['bottom',[0,0,0,0]],['release15',[Math.PI+Math.PI/12,0,0,0]],['drop',[Math.PI+Math.PI/180,0,0,0]],['kick4',[0,0,0,0]]];
const evaluations=[];
for(const [name,start] of cases){let s=start,hold=0,preKick=false;for(let t=0;t<2400;t++){
 if(name==='kick4'&&t===800){preKick=hold>=10;s=[s[0],s[1],s[2]+4,s[3]-4];hold=0}
 const r=C.step(model,s,!(name==='drop'&&t<40));s=r.state;hold=r.allBalanced?hold+.05:r.holdTail;
}evaluations.push({name,success:hold>=10,final_hold:hold,preKick,state:s})}
process.stdout.write(JSON.stringify({torques,transitions,evaluations}));
