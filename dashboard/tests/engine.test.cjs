'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const engine=require('../dist/engine.js');
const close=(a,b,tolerance=1e-11)=>assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
const bars=engine.generateDemo(42,760);
const result=engine.runExperiment(bars,{synthetic:true});
const csv=(rows,header='Date,Open,High,Low,Close,Volume')=>header+'\n'+rows.map(b=>[b.date,b.open,b.high,b.low,b.close,b.volume].join(',')).join('\n');

test('engine is deterministic and browser file compatible',()=>{
  assert.deepEqual(engine.generateDemo(42,180),engine.generateDemo(42,180));
  const context=vm.createContext({});
  vm.runInContext(fs.readFileSync(require.resolve('../dist/engine.js'),'utf8'),context);
  assert.equal(typeof context.SignalEngine.runExperiment,'function');
  assert.equal(context.SignalEngine.generateDemo(42,180).length,180);
});

test('CSV supports case-insensitive headers, quotes, BOM, CRLF, and chronological sorting',()=>{
  const input='\uFEFF"dAtE","OPEN","HIGH","LOW","Close","Volume"\r\n'+bars.slice(0,180).reverse().map(b=>[b.date,b.open,b.high,b.low,b.close,b.volume].map(v=>'"'+v+'"').join(',')).join('\r\n');
  const parsed=engine.parseCSV(input);
  assert.deepEqual(parsed.bars,bars.slice(0,180));
  assert.ok(parsed.warnings.some(w=>w.includes('sorted')));
  const minimal=engine.parseCSV('Date,Open,Close\n'+bars.slice(0,180).map(b=>`${b.date},${b.open},${b.close}`).join('\n'));
  assert.equal(minimal.bars[0].volume,0);
  assert.equal(minimal.bars[0].high,Math.max(bars[0].open,bars[0].close));
  assert.equal(minimal.warnings.length,2);
});

test('CSV rejects malformed, duplicate, multi-ticker, invalid, and undersized data',()=>{
  assert.throws(()=>engine.parseCSV('<html>error</html>'),/HTML/);
  assert.throws(()=>engine.parseCSV('Date,Close\n2025-01-01,1'),/Open|open/);
  assert.throws(()=>engine.parseCSV(csv(bars.slice(0,179))),/180/);
  assert.throws(()=>engine.parseCSV(csv([...bars.slice(0,180),bars[0]])),/Duplicate/);
  assert.throws(()=>engine.parseCSV(csv(bars.slice(0,180)).replace(bars[0].date,'2022-02-30')),/calendar/);
  assert.throws(()=>engine.parseCSV(csv(bars.slice(0,180)).replace(String(bars[0].open),'Infinity')),/finite/);
  assert.throws(()=>engine.parseCSV(csv(bars.slice(0,180))+'\n"unclosed'),/quotation/);
  assert.throws(()=>engine.parseCSV('Date,Open,Close,Symbol\n'+bars.slice(0,180).map((b,i)=>`${b.date},${b.open},${b.close},${i%2?'A':'B'}`).join('\n')),/Multiple tickers/);
  assert.throws(()=>engine.parseCSV(csv(bars.slice(0,180)).replace(String(bars[0].high),'0')),/High/);
});

test('CSV requires nonempty consistent Symbol/Ticker values and column names',()=>{
  const stockRows=bars.slice(0,180);
  const withSymbols=(fn)=>'Date,Open,Close,Symbol,Ticker\n'+stockRows.map((b,i)=>`${b.date},${b.open},${b.close},${fn(i)}`).join('\n');
  assert.equal(engine.parseCSV(withSymbols(()=> ' abc ,ABC')).bars.length,180);
  assert.throws(()=>engine.parseCSV(withSymbols(()=> 'ABC,XYZ')),/disagree/);
  assert.throws(()=>engine.parseCSV(withSymbols(i=>i===12?'ABC,   ':'ABC,ABC')),/must not be empty/);
  assert.throws(()=>engine.parseCSV('Date,Open,Close,Symbol\n'+stockRows.map((b,i)=>`${b.date},${b.open},${b.close},${i===7?'':'ABC'}`).join('\n')),/must not be empty/);
  assert.throws(()=>engine.parseCSV('Date,Open,Close,   \n'+stockRows.map(b=>`${b.date},${b.open},${b.close},ABC`).join('\n')),/column names must not be empty/);
});

test('split purges, label maturity, scaler data, and validation selection have actual evidence',()=>{
  assert.equal(result.validation.purgeBars,2);
  assert.ok(result.meta.trainRange[1]<result.meta.validationRange[0]);
  assert.ok(result.meta.validationRange[1]<result.meta.testRange[0]);
  assert.ok(result.audit.every(a=>a.passed));
  for(const fit of result.validation.trainingEvents){
    assert.ok(fit.lastTrainingFeatureDate<fit.maxTrainingLabelDate);
    assert.ok(fit.maxTrainingLabelDate<fit.signalDate);
    assert.equal(fit.scalerMeans.length,9);
    assert.ok(fit.scalerStandardDeviations.every(s=>s>0&&Number.isFinite(s)));
  }
  const trainStart=bars.findIndex(b=>b.date===result.meta.trainRange[0]);
  const trainEnd=bars.findIndex(b=>b.date===result.meta.trainRange[1]);
  let actualMean=0;
  for(let i=trainStart;i<=trainEnd;i++)actualMean+=(bars[i].close/bars[i-1].close-1)/(trainEnd-trainStart+1);
  close(actualMean,result.validation.trainingEvents[0].scalerMeans[0]);
  assert.ok(result.validation.selection.maxValidationLabelDate<result.validation.selection.firstTestSignalDate);
  close(result.models.reduce((s,m)=>s+m.weight,0),1);
  for(let m=0;m<3;m++){
    const inverse=result.models.map(x=>1/x.validationBrier);
    close(result.models[m].weight,inverse[m]/inverse.reduce((a,b)=>a+b,0));
  }
});

test('future price changes cannot alter earlier test predictions or validation weights',()=>{
  const cutoff=690;
  const changed=bars.map((b,i)=>i<cutoff?{...b}:{...b,open:b.open*2,close:b.close*2,high:b.high*2,low:b.low*2,volume:b.volume*5});
  const rerun=engine.runExperiment(changed,{synthetic:true});
  assert.deepEqual(rerun.validation.weights,result.validation.weights);
  const earlier=result.predictions.filter(p=>p.signalDate<bars[cutoff].date);
  for(let j=0;j<earlier.length;j++){
    assert.equal(rerun.predictions[j].probability,earlier[j].probability);
    assert.equal(rerun.predictions[j].position,earlier[j].position);
  }
  assert.ok(earlier.length>50);
});

test('labels and holdings align exactly with next-open to following-open returns',()=>{
  for(const p of result.predictions){
    const i=bars.findIndex(b=>b.date===p.signalDate);
    assert.equal(p.executionDate,bars[i+1].date);
    assert.equal(p.exitDate,bars[i+2].date);
    close(p.actualReturn,bars[i+2].open/bars[i+1].open-1);
    assert.ok(p.position>=0&&p.position<=1);
  }
  const first=result.predictions[0],i=bars.findIndex(b=>b.date===first.signalDate),changed=bars.map(b=>({...b}));
  changed[i+1].open*=1.05;changed[i+1].high=Math.max(changed[i+1].high,changed[i+1].open);
  const rerun=engine.runExperiment(changed);
  assert.equal(rerun.predictions[0].probability,first.probability);
  close(rerun.predictions[0].actualReturn,changed[i+2].open/changed[i+1].open-1);
  assert.notEqual(rerun.predictions[0].actualReturn,first.actualReturn);
});

test('equity charges drift-aware allocation-change costs and final liquidation to both strategies',()=>{
  const fee=result.options.feeBps/10000;
  let equity=result.options.initialCapital,previous=0,tradeCount=0;
  for(let j=0;j<result.predictions.length;j++){
    const p=result.predictions[j],change=Math.abs(p.position-previous);
    equity*=1-fee*change;
    equity*=1+p.position*p.actualReturn;
    const drifted=p.position*(1+p.actualReturn)/(1+p.position*p.actualReturn);
    if(change>1e-10)tradeCount++;
    if(j===result.predictions.length-1&&drifted>1e-10){equity*=1-fee*drifted;tradeCount++;}
    close(equity,result.series[j].equity,1e-8);
    previous=drifted;
  }
  close(equity,result.metrics.endingEquity,1e-8);
  assert.equal(result.metrics.tradeCount,tradeCount);
  let baseline=result.options.initialCapital*(1-fee);
  for(const p of result.predictions)baseline*=1+p.actualReturn;
  baseline*=1-fee;
  close(baseline,result.baselineMetrics.endingEquity,1e-8);
  assert.equal(result.baselineMetrics.tradeCount,2);
});

test('constant fractional targets rebalance after price drift and liquidate drifted holdings',()=>{
  // Instrument the private pure simulator in a VM; the shipped public API stays unchanged.
  const context=vm.createContext({});
  const source=fs.readFileSync(require.resolve('../dist/engine.js'),'utf8');
  vm.runInContext(source.replace("return {generateDemo,parseCSV,runExperiment,version:'1.0.0'};","return {generateDemo,parseCSV,runExperiment,version:'1.0.0',portfolioForTest:portfolio};"),context);
  const prices=[100,100,110,121].map((open,i)=>({date:`2025-01-0${i+1}`,open}));
  const simulated=context.SignalEngine.portfolioForTest([{i:0,return:.1,y:1},{i:1,return:.1,y:1}],[.7,.7],[.5,.5],prices,{initialCapital:10000,feeBps:100});
  const drifted=.5*1.1/1.05;
  const expected=10000*(1-.01*.5)*1.05*(1-.01*Math.abs(.5-drifted))*1.05*(1-.01*drifted);
  close(simulated.metrics.endingEquity,expected,1e-8);
  assert.equal(simulated.metrics.tradeCount,3);
  assert.equal(simulated.trades[1].action,'SELL');
  assert.equal(simulated.trades[1].weight,.5);
  assert.equal(simulated.trades[2].action,'LIQUIDATE');
  close(simulated.trades[2].netReturn,-.01*drifted);
});

test('cash-only and zero-cost cases have correct arithmetic',()=>{
  const cash=engine.runExperiment(bars,{maxExposure:0,feeBps:100});
  assert.equal(cash.metrics.endingEquity,10000);
  assert.equal(cash.metrics.totalReturn,0);
  assert.equal(cash.metrics.maxDrawdown,0);
  assert.equal(cash.metrics.tradeCount,0);
  assert.equal(cash.latest.signal,'CASH');
  const free=engine.runExperiment(bars,{feeBps:0});
  let baseline=10000;for(const p of free.predictions)baseline*=1+p.actualReturn;
  close(free.baselineMetrics.endingEquity,baseline,1e-8);
});

test('latest signal is inference on final close without an invented realized outcome',()=>{
  assert.equal(result.latest.date,bars[bars.length-1].date);
  assert.ok(!Object.hasOwn(result.latest,'actualReturn'));
  assert.ok(result.predictions.every(p=>p.signalDate<result.latest.date));
  assert.equal(result.predictions[result.predictions.length-1].exitDate,result.latest.date);
  assert.equal(result.validation.probabilitiesCalibrated,false);
  close(result.features.reduce((s,f)=>s+f.importance,0),1);
});

test('options reject nonfinite and prohibited exposures',()=>{
  assert.throws(()=>engine.runExperiment(bars,{feeBps:NaN}),/finite/);
  assert.throws(()=>engine.runExperiment(bars,{initialCapital:Infinity}),/finite/);
  assert.throws(()=>engine.runExperiment(bars,{maxExposure:1.01}),/leverage/);
  assert.throws(()=>engine.runExperiment(bars,{retrainEvery:0}),/interval/);
  assert.throws(()=>engine.runExperiment(bars,{threshold:.99}),/threshold/);
});
