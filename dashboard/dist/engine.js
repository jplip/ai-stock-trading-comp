/* Signal Lab: dependency-free, chronological educational research engine. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.SignalEngine = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const YEAR = 252, LOOKBACK = 20;
  const DEFAULTS = { initialCapital: 10000, feeBps: 10, threshold: .55, maxExposure: 1,
    targetVol: .15, trainFraction: .6, validationFraction: .2, retrainEvery: 20,
    sourceName: 'Uploaded CSV', synthetic: false };
  const FEATURE_INFO = [
    ['1-day return', 'Latest close divided by the previous close, minus one.'],
    ['5-day return', 'Latest close divided by the close five sessions earlier, minus one.'],
    ['10-day return', 'Latest close divided by the close ten sessions earlier, minus one.'],
    ['20-day trend', 'Latest close relative to the trailing 20-session average.'],
    ['20-day volatility', 'Standard deviation of the last 20 daily close returns; daily units.'],
    ['Daily range', 'Today\'s high minus low, divided by today\'s close.'],
    ['Opening gap', 'Today\'s open relative to the previous close.'],
    ['Intraday return', 'Today\'s close relative to today\'s open.'],
    ['Relative volume', 'Today\'s volume relative to its trailing 20-session average; zero when unavailable.']
  ];
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const mean = a => a.reduce((s, x) => s + x, 0) / (a.length || 1);
  function deviation(a) { const m = mean(a); return Math.sqrt(a.reduce((s,x) => s+(x-m)*(x-m),0) / Math.max(1,a.length-1)); }
  function random(seed) { let n = seed >>> 0; return () => { n += 0x6D2B79F5; let t = n; t = Math.imul(t ^ t >>> 15,t | 1); t ^= t + Math.imul(t ^ t >>> 7,t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function canonicalDate(value) {
    const s = String(value).trim();
    let y,m,d, hit = /^(\d{4})-(\d{1,2})-(\d{1,2})(?:T(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d+)?)?(?:Z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)?)?$/.exec(s);
    if (hit) { y=+hit[1]; m=+hit[2]; d=+hit[3]; }
    else { hit = /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/.exec(s); if (hit) { y=+hit[3]; m=+hit[1]; d=+hit[2]; } }
    if (!hit || y < 1900 || y > 2200) throw new Error('Invalid date "'+s+'". Use YYYY-MM-DD or M/D/YYYY.');
    const t = new Date(Date.UTC(y,m-1,d));
    if (t.getUTCFullYear()!==y || t.getUTCMonth()!==m-1 || t.getUTCDate()!==d) throw new Error('Invalid calendar date "'+s+'".');
    return t.toISOString().slice(0,10);
  }
  function normalizeBars(input) {
    if (!Array.isArray(input) || input.length < 180) throw new Error('At least 180 daily price rows are required.');
    if (input.length > 5000) throw new Error('Use at most 5,000 daily rows for this browser demo.');
    const dates=new Set(), warnings=[];
    const bars=input.map((b,i) => {
      const date=canonicalDate(b.date), open=Number(b.open), close=Number(b.close);
      if (dates.has(date)) throw new Error('Duplicate date '+date+'. Supply one stock with one row per date.');
      dates.add(date);
      if (!(open>0 && close>0 && Number.isFinite(open) && Number.isFinite(close))) throw new Error('Row '+(i+1)+': Open and Close must be finite positive prices.');
      const high=b.high==null || b.high==='' ? Math.max(open,close) : Number(b.high);
      const low=b.low==null || b.low==='' ? Math.min(open,close) : Number(b.low);
      const volume=b.volume==null || b.volume==='' ? 0 : Number(b.volume);
      if (![high,low,volume].every(Number.isFinite) || low<=0 || high<Math.max(open,close) || low>Math.min(open,close) || volume<0) throw new Error('Row '+(i+1)+': invalid High, Low, or Volume. High/Low must contain Open and Close.');
      return {date,open,high,low,close,volume};
    });
    if (bars.some((b,i)=>i && b.date<bars[i-1].date)) { bars.sort((a,b)=>a.date.localeCompare(b.date)); warnings.push('Dates were not chronological. Rows have been sorted from oldest to newest.'); }
    return {bars,warnings};
  }
  function csvRows(text) {
    const rows=[]; let row=[],cell='',quoted=false;
    for(let i=0;i<text.length;i++) {
      const c=text[i];
      if(c==='"') { if(quoted && text[i+1]==='"') {cell+='"';i++;} else if(quoted || cell==='') quoted=!quoted; else throw new Error('Malformed CSV quotation.'); }
      else if(c===',' && !quoted) {row.push(cell);cell='';}
      else if((c==='\n'||c==='\r') && !quoted) { if(c==='\r'&&text[i+1]==='\n')i++; row.push(cell); if(row.some(v=>v.trim()!==''))rows.push(row); row=[];cell=''; }
      else cell+=c;
    }
    if(quoted)throw new Error('CSV contains an unclosed quotation.');
    row.push(cell); if(row.some(v=>v.trim()!==''))rows.push(row);
    return rows;
  }
  function parseCSV(text) {
    if(typeof text!=='string'||!text.trim())throw new Error('The CSV is empty.');
    if(/^\s*</.test(text))throw new Error('This looks like HTML. Export a price table as CSV.');
    const rows=csvRows(text.replace(/^\uFEFF/,''));
    if(!rows.length)throw new Error('The CSV is empty.');
    const header=rows.shift().map(s=>s.trim().toLowerCase());
    if(header.some(s=>s===''))throw new Error('CSV column names must not be empty.');
    if(new Set(header).size!==header.length)throw new Error('CSV column names must be unique.');
    for(const key of ['date','open','close'])if(!header.includes(key))throw new Error('Missing required column: '+key+'. Required: Date, Open, Close.');
    const index=Object.fromEntries(header.map((v,i)=>[v,i]));
    const symbols=new Set(); let missingRange=false,missingVolume=false;
    const input=rows.map((r,i)=>{
      if(r.length!==header.length)throw new Error('CSV line '+(i+2)+' has '+r.length+' fields; expected '+header.length+'.');
      const rowSymbols=[];
      for(const key of ['symbol','ticker'])if(index[key]!==undefined){
        const symbol=r[index[key]].trim().toUpperCase();
        if(!symbol)throw new Error('CSV line '+(i+2)+': '+key+' must not be empty when its column is present.');
        rowSymbols.push(symbol);symbols.add(symbol);
      }
      if(rowSymbols.length===2&&rowSymbols[0]!==rowSymbols[1])throw new Error('CSV line '+(i+2)+': Symbol and Ticker disagree. Upload one stock at a time.');
      const b={date:r[index.date],open:r[index.open].trim(),close:r[index.close].trim()};
      for(const k of ['high','low','volume']) { const v=index[k]===undefined?'':r[index[k]].trim(); b[k]=v; if(v==='') {if(k==='volume')missingVolume=true;else missingRange=true;} }
      return b;
    });
    if(symbols.size>1)throw new Error('Multiple tickers found. Upload one stock at a time.');
    const result=normalizeBars(input);
    if(missingRange)result.warnings.push('Missing High/Low values were filled from Open and Close; the range feature is approximate.');
    if(missingVolume)result.warnings.push('Missing Volume values were set to zero; relative volume is unavailable or incomplete.');
    return result;
  }
  function generateDemo(seed=42, days=760) {
    if(!Number.isInteger(days)||days<180||days>5000)throw new Error('Demo length must be an integer from 180 to 5,000.');
    if(!Number.isFinite(seed))throw new Error('Demo seed must be finite.');
    const rng=random(seed), gaussian=()=>Math.sqrt(-2*Math.log(Math.max(1e-12,rng())))*Math.cos(2*Math.PI*rng());
    const bars=[]; let close=100, date=new Date(Date.UTC(2022,0,3));
    for(let i=0;i<days;i++) {
      while(date.getUTCDay()===0 || date.getUTCDay()===6)date.setUTCDate(date.getUTCDate()+1);
      const open=close*Math.exp(gaussian()*.004), daily=gaussian()*.012+.0002;
      close=open*Math.exp(daily);
      const high=Math.max(open,close)*Math.exp(Math.abs(gaussian())*.005);
      const low=Math.min(open,close)*Math.exp(-Math.abs(gaussian())*.005);
      bars.push({date:date.toISOString().slice(0,10),open,high,low,close,volume:Math.round(900000*Math.exp(gaussian()*.3))});
      date.setUTCDate(date.getUTCDate()+1);
    }
    return bars;
  }
  function featuresAt(bars,i) {
    const b=bars[i], closes=bars.slice(i-19,i+1).map(x=>x.close), volumes=bars.slice(i-19,i+1).map(x=>x.volume);
    const returns=[]; for(let k=i-19;k<=i;k++)returns.push(bars[k].close/bars[k-1].close-1);
    const vol=deviation(returns), avgVolume=mean(volumes);
    return {i,x:[b.close/bars[i-1].close-1,b.close/bars[i-5].close-1,b.close/bars[i-10].close-1,b.close/mean(closes)-1,vol,(b.high-b.low)/b.close,b.open/bars[i-1].close-1,b.close/b.open-1,avgVolume>0?b.volume/avgVolume-1:0],volatility:vol*Math.sqrt(YEAR)};
  }
  function fitScaler(rows) {
    const d=FEATURE_INFO.length,mu=Array(d).fill(0),sd=Array(d).fill(0);
    for(const r of rows)for(let j=0;j<d;j++)mu[j]+=r.x[j]/rows.length;
    for(const r of rows)for(let j=0;j<d;j++)sd[j]+=(r.x[j]-mu[j])**2/rows.length;
    for(let j=0;j<d;j++)sd[j]=Math.sqrt(sd[j])||1;
    return {mu,sd,transform:x=>x.map((v,j)=>clamp((v-mu[j])/sd[j],-12,12))};
  }
  const sigmoid=z=>1/(1+Math.exp(-clamp(z,-30,30)));
  function fitLogistic(xs,ys) {
    const d=xs[0].length,w=Array(d).fill(0); let intercept=Math.log((ys.reduce((a,b)=>a+b,0)+1)/(ys.length-ys.reduce((a,b)=>a+b,0)+1));
    for(let epoch=0;epoch<150;epoch++) {
      const g=Array(d).fill(0); let gb=0;
      for(let i=0;i<xs.length;i++) { let z=intercept;for(let j=0;j<d;j++)z+=w[j]*xs[i][j];const error=sigmoid(z)-ys[i];gb+=error;for(let j=0;j<d;j++)g[j]+=error*xs[i][j]; }
      intercept-=.12*gb/xs.length;
      for(let j=0;j<d;j++)w[j]-=.12*(g[j]/xs.length+.035*w[j]);
    }
    return {coefficients:w,predict:x=>sigmoid(intercept+w.reduce((s,v,j)=>s+v*x[j],0))};
  }
  function fitForest(xs,ys,seed) {
    const rng=random(seed),d=xs[0].length,minLeaf=7;
    function tree(ids,depth) {
      const count=ids.reduce((s,i)=>s+ys[i],0),probability=(count+1)/(ids.length+2);
      if(depth>=4||ids.length<minLeaf*2||count===0||count===ids.length)return {probability};
      let best=null,bestLoss=Infinity;
      const candidates=[];while(candidates.length<3){const f=Math.floor(rng()*d);if(!candidates.includes(f))candidates.push(f);}
      for(const f of candidates)for(let trial=0;trial<5;trial++) {
        const threshold=xs[ids[Math.floor(rng()*ids.length)]][f];let ln=0,ly=0,rn=0,ry=0;
        for(const i of ids) {if(xs[i][f]<=threshold){ln++;ly+=ys[i];}else{rn++;ry+=ys[i];}}
        if(ln<minLeaf||rn<minLeaf)continue;
        const loss=ly*(1-ly/ln)+ry*(1-ry/rn);
        if(loss<bestLoss){bestLoss=loss;best={f,threshold};}
      }
      if(!best)return {probability};
      const left=[],right=[];for(const i of ids)(xs[i][best.f]<=best.threshold?left:right).push(i);
      return {f:best.f,threshold:best.threshold,left:tree(left,depth+1),right:tree(right,depth+1),probability};
    }
    const trees=[];for(let t=0;t<17;t++){const ids=[];for(let j=0;j<Math.min(480,xs.length);j++)ids.push(Math.floor(rng()*xs.length));trees.push(tree(ids,0));}
    return {predict:x=>mean(trees.map(node=>{while(node.left)node=x[node.f]<=node.threshold?node.left:node.right;return node.probability;}))};
  }
  function fitKNN(xs,ys) {
    const k=Math.min(25,Math.max(7,Math.round(Math.sqrt(xs.length))));
    return {predict:x=>{const nearest=[];for(let i=0;i<xs.length;i++){let distance=0;for(let j=0;j<x.length;j++)distance+=(x[j]-xs[i][j])**2;if(nearest.length<k||distance<nearest[nearest.length-1][0]){let at=nearest.length;while(at>0&&nearest[at-1][0]>distance)at--;nearest.splice(at,0,[distance,ys[i]]);if(nearest.length>k)nearest.pop();}}let numerator=1,denominator=2;for(const [distance,y] of nearest){const weight=1/(.3+Math.sqrt(distance));numerator+=weight*y;denominator+=weight;}return numerator/denominator;}};
  }
  function fitModels(rows,seed) {
    const scaler=fitScaler(rows),xs=rows.map(r=>scaler.transform(r.x)),ys=rows.map(r=>r.y);
    const models=[fitLogistic(xs,ys),fitForest(xs,ys,seed),fitKNN(xs,ys)];
    return {scaler,models,predict:x=>{const z=scaler.transform(x);return models.map(m=>m.predict(z));}};
  }
  function validateOptions(opts) {
    const o=Object.assign({},DEFAULTS,opts);
    for(const key of ['initialCapital','feeBps','threshold','maxExposure','targetVol','trainFraction','validationFraction','retrainEvery'])if(typeof o[key]!=='number'||!Number.isFinite(o[key]))throw new Error(key+' must be a finite number.');
    if(o.initialCapital<=0||o.initialCapital>1e12)throw new Error('Initial capital must be positive and at most 1 trillion.');
    if(o.feeBps<0||o.feeBps>100)throw new Error('One-way costs must be between 0 and 100 basis points.');
    if(o.threshold<.5||o.threshold>=.95)throw new Error('Probability threshold must be at least 0.50 and below 0.95.');
    if(o.maxExposure<0||o.maxExposure>1)throw new Error('Maximum exposure must be between 0 and 1 (no leverage).');
    if(o.targetVol<=0||o.targetVol>1)throw new Error('Annual volatility target must be above 0 and at most 1.');
    if(o.trainFraction<.4||o.validationFraction<.1||o.trainFraction+o.validationFraction>.9)throw new Error('Use at least 40% training, 10% validation, and 10% testing.');
    if(!Number.isInteger(o.retrainEvery)||o.retrainEvery<1||o.retrainEvery>250)throw new Error('Retrain interval must be an integer from 1 to 250.');
    o.synthetic=Boolean(o.synthetic);o.sourceName=String(o.sourceName).replace(/[\u0000-\u001f<>]/g,'').slice(0,160);
    return o;
  }
  function positionFor(probability,volatility,o) {
    if(probability<=o.threshold)return 0;
    const conviction=clamp((probability-o.threshold)/(1-o.threshold),0,1);
    return Math.min(o.maxExposure,o.targetVol/Math.max(.01,volatility))*conviction;
  }
  function portfolio(rows,probabilities,positions,bars,o) {
    const fee=o.feeBps/10000,series=[],trades=[],daily=[];let equity=o.initialCapital,peak=equity,driftedPosition=0,drawdown=0,tradeCount=0;
    for(let j=0;j<rows.length;j++) {
      const r=rows[j],p=positions[j],delta=p-driftedPosition,before=equity,cost=fee*Math.abs(delta);
      equity*=1-cost;equity*=1+p*r.return;
      driftedPosition=p*(1+r.return)/(1+p*r.return);
      const periodEquity=equity;
      if(Math.abs(delta)>1e-10){tradeCount++;trades.push({signalDate:bars[r.i].date,executionDate:bars[r.i+1].date,exitDate:bars[r.i+2].date,action:delta>0?'BUY':'SELL',weight:p,probability:probabilities[j],netReturn:periodEquity/before-1,equity:periodEquity});}
      if(j===rows.length-1&&driftedPosition>1e-10){equity*=1-fee*driftedPosition;tradeCount++;trades.push({signalDate:bars[r.i].date,executionDate:bars[r.i+2].date,exitDate:bars[r.i+2].date,action:'LIQUIDATE',weight:0,probability:probabilities[j],netReturn:-fee*driftedPosition,equity});}
      const ret=equity/before-1;daily.push(ret);peak=Math.max(peak,equity);const dd=1-equity/peak;drawdown=Math.max(drawdown,dd);
      series.push({date:bars[r.i+2].date,equity,drawdown:dd,position:p,probability:probabilities[j],price:bars[r.i+2].open,return:ret});
    }
    const sd=deviation(daily);
    return {series,trades,metrics:{totalReturn:equity/o.initialCapital-1,maxDrawdown:drawdown,sharpe:sd>1e-12?mean(daily)/sd*Math.sqrt(YEAR):0,accuracy:mean(rows.map((r,j)=>Number((probabilities[j]>=.5)===Boolean(r.y)))),tradeCount,exposure:mean(positions),endingEquity:equity}};
  }
  function runExperiment(input,opts={}) {
    const started=Date.now(),o=validateOptions(opts),normalized=normalizeBars(input),bars=normalized.bars,all=[];
    for(let i=LOOKBACK;i<bars.length-2;i++){const row=featuresAt(bars,i);row.return=bars[i+2].open/bars[i+1].open-1;row.y=Number(row.return>0);all.push(row);}
    const trainBoundary=Math.floor(all.length*o.trainFraction),testBoundary=Math.floor(all.length*(o.trainFraction+o.validationFraction));
    const train=all.slice(0,trainBoundary-2),validation=all.slice(trainBoundary,testBoundary-2),test=all.slice(testBoundary);
    if(train.length<60||validation.length<10||test.length<10)throw new Error('This split leaves too little data. Add more daily rows or use the default 60/20/20 split.');
    const fitLog=[];
    function fitted(rows,at,phase) {
      const fit=fitModels(rows,16771+at.i);
      fitLog.push({phase,featureDate:bars[at.i].date,lastTrainingFeature:bars[rows[rows.length-1].i].date,lastTrainingLabelExit:bars[rows[rows.length-1].i+2].date,count:rows.length,scalerMeans:fit.scaler.mu.slice(),scalerStandardDeviations:fit.scaler.sd.slice()});
      return fit;
    }
    const initial=fitted(train,validation[0],'validation'),validationPredictions=validation.map(r=>initial.predict(r.x));
    const briers=[0,1,2].map(m=>mean(validation.map((r,j)=>(validationPredictions[j][m]-r.y)**2)));
    const rawWeights=briers.map(e=>1/Math.max(.000001,e)),sumWeight=rawWeights.reduce((a,b)=>a+b,0),weights=rawWeights.map(w=>w/sumWeight);
    let current=null;const probabilities=[],perModel=[[],[],[]],positions=[];
    for(let j=0;j<test.length;j++) {
      const r=test[j];
      if(j%o.retrainEvery===0)current=fitted(all.filter(t=>t.i<=r.i-3),r,'test');
      const p=current.predict(r.x);for(let m=0;m<3;m++)perModel[m].push(p[m]);const probability=p.reduce((s,v,m)=>s+v*weights[m],0);
      probabilities.push(probability);positions.push(positionFor(probability,r.volatility,o));
    }
    const latestFeature=featuresAt(bars,bars.length-1),latestFit=fitted(all.filter(t=>t.i<=latestFeature.i-3),latestFeature,'latest');
    const latestModelProbabilities=latestFit.predict(latestFeature.x),latestProbability=latestModelProbabilities.reduce((s,p,m)=>s+p*weights[m],0);
    const strategy=portfolio(test,probabilities,positions,bars,o),baseline=portfolio(test,test.map(()=>1),test.map(()=>1),bars,o);
    const models=[
      {id:'logistic',name:'Logistic regression',description:'Regularized linear classifier trained with gradient descent; sees standardized historical features.'},
      {id:'forest',name:'Random forest',description:'17 seeded bootstrap decision trees, maximum depth 4; averages learned leaf probabilities.'},
      {id:'knn',name:'Nearest neighbors',description:'Distance-weighted votes from up to 25 similar historical observations, using training-only scaling.'}
    ].map((m,j)=>Object.assign(m,{weight:weights[j],validationBrier:briers[j],testAccuracy:mean(test.map((r,k)=>Number((perModel[j][k]>=.5)===Boolean(r.y)))),latestProbability:latestModelProbabilities[j]}));
    const coefficientMagnitude=latestFit.models[0].coefficients.map(Math.abs),coefficientSum=coefficientMagnitude.reduce((a,b)=>a+b,0)||1;
    const features=FEATURE_INFO.map((f,j)=>({name:f[0],value:latestFeature.x[j],importance:coefficientMagnitude[j]/coefficientSum,coefficient:latestFit.models[0].coefficients[j],description:f[1]+' Importance is absolute standardized logistic coefficient share, not ensemble causality.'}));
    const predictions=test.map((r,j)=>({signalDate:bars[r.i].date,executionDate:bars[r.i+1].date,exitDate:bars[r.i+2].date,probability:probabilities[j],position:positions[j],actualReturn:r.return}));
    const firstTest=test[0],lastValidation=validation[validation.length-1],matured=fitLog.every(f=>f.lastTrainingLabelExit<f.featureDate);
    const audit=[
      {label:'Chronological split',passed:train[train.length-1].i<validation[0].i&&lastValidation.i<firstTest.i,detail:train.length+' train / '+validation.length+' validation / '+test.length+' test observations; oldest to newest.'},
      {label:'Two-bar boundary purge',passed:validation[0].i-train[train.length-1].i===3&&firstTest.i-lastValidation.i===3,detail:'The final two feature rows of training and validation are excluded from those splits.'},
      {label:'Matured training labels',passed:matured,detail:fitLog.length+' fits audited. Every fitted label exits before its prediction feature date; '+fitLog[0].lastTrainingLabelExit+' < '+fitLog[0].featureDate+' at the first fit.'},
      {label:'Training-only scaling',passed:true,detail:'Means and standard deviations are recomputed only from each fit\'s eligible training rows. Validation and current/future test rows are not passed to the scaler.'},
      {label:'Validation-only ensemble weights',passed:bars[lastValidation.i+2].date<bars[firstTest.i].date,detail:'Inverse Brier weights use '+validation.length+' validation labels only. Last validation outcome: '+bars[lastValidation.i+2].date+'; first test signal: '+bars[firstTest.i].date+'.'},
      {label:'Untouched test selection',passed:true,detail:'Threshold and other settings are user inputs; model specifications are fixed. Test outcomes only enter later scheduled training once matured. Changing settings after viewing results is further test tuning.'},
      {label:'Next-open execution',passed:predictions.every(p=>p.signalDate<p.executionDate&&p.executionDate<p.exitDate),detail:'Features at close t → enter at Open t+1 → value/exit at Open t+2. Holdings form non-overlapping one-session intervals.'},
      {label:'Costs and liquidation',passed:true,detail:o.feeBps+' bps per absolute change from drifted pre-trade allocation to target allocation; terminal liquidation uses the final drifted weight. Both strategies pay entry and liquidation costs. Multiplicative allocation-cost approximation; no additional slippage.'},
      {label:'Exposure limits',passed:positions.every(p=>p>=0&&p<=o.maxExposure+1e-12),detail:'Long/cash only. Conviction-scaled allocation with an annualized trailing-volatility target and '+Math.round(o.maxExposure*100)+'% cap. Cash earns zero.'},
      {label:'Latest signal has no future outcome',passed:true,detail:'Latest close '+bars[latestFeature.i].date+' is inference only. Its next-open return is unknown and excluded from test results.'},
      {label:'Uncalibrated probabilities',passed:true,detail:'Model outputs and their ensemble are uncalibrated scores; they are not verified real-world success probabilities.'}
    ];
    if(o.synthetic)audit.push({label:'Synthetic demonstration',passed:true,detail:'Generated geometric random-walk OHLCV data; no real ticker or live market feed. Generator is not trained or tuned to reward the strategy.'});
    return {
      meta:{sourceName:o.sourceName,rows:bars.length,trainRange:[bars[train[0].i].date,bars[train[train.length-1].i].date],validationRange:[bars[validation[0].i].date,bars[lastValidation.i].date],testRange:[bars[firstTest.i].date,bars[test[test.length-1].i+2].date],trainCount:train.length,validationCount:validation.length,testCount:test.length,synthetic:o.synthetic,timing:'Close t features → next Open t+1 entry → Open t+2 outcome',warnings:normalized.warnings,elapsedMs:Date.now()-started},
      metrics:strategy.metrics,baselineMetrics:baseline.metrics,
      series:strategy.series.map((r,j)=>Object.assign({},r,{baseline:baseline.series[j].equity})),
      models,features,latest:{date:bars[latestFeature.i].date,probability:latestProbability,position:positionFor(latestProbability,latestFeature.volatility,o),signal:latestProbability>o.threshold&&o.maxExposure>0?'LONG':'CASH',disagreement:Math.max(...latestModelProbabilities)-Math.min(...latestModelProbabilities),volatility:latestFeature.volatility},
      trades:strategy.trades,predictions,audit,options:o,
      validation:{brierByModel:briers,weights,ensembleBrier:mean(validation.map((r,j)=>(validationPredictions[j].reduce((s,p,m)=>s+p*weights[m],0)-r.y)**2)),weightMethod:'Normalized inverse validation Brier error',probabilitiesCalibrated:false,purgeBars:2,fitLog,trainingEvents:fitLog.map(f=>({phase:f.phase,signalDate:f.featureDate,maxTrainingLabelDate:f.lastTrainingLabelExit,lastTrainingFeatureDate:f.lastTrainingFeature,trainingCount:f.count,scalerMeans:f.scalerMeans,scalerStandardDeviations:f.scalerStandardDeviations})),selection:{maxValidationLabelDate:bars[lastValidation.i+2].date,firstTestSignalDate:bars[firstTest.i].date,weightsFixedBeforeTest:true},featureNames:FEATURE_INFO.map(f=>f[0]),label:'Open[t+2] / Open[t+1] - 1',lookback:LOOKBACK}
    };
  }
  return {generateDemo,parseCSV,runExperiment,version:'1.0.0'};
});
