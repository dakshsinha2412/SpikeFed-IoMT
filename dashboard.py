"""Premium local control surface for the existing federated SNN pipeline."""
from __future__ import annotations
import json, subprocess, sys, time, zipfile, re
from datetime import datetime
from pathlib import Path
import streamlit as st
from model import SpikingCNN, parameter_count

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/'results'; CHECKPOINT=ROOT/'checkpoints'/'global_model.pt'; UPLOADS=ROOT/'uploaded_datasets'
NAV=['Overview','Experiments','Federated Learning','Hospital Clients','Model Architecture','Dataset Management','Performance Analytics','Experiment History','Model Checkpoints','System Logs','Raspberry Pi Deployment','Settings']

def files(): return sorted([p for p in RESULTS.glob('*_results.json') if not p.name.startswith('test_')],key=lambda p:p.stat().st_mtime,reverse=True)
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def latest(): return load(files()[0]) if files() else None
def css():
 st.markdown('''<style>
 @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
 .stApp{background:#0a0c11;color:#e9edf5;font-family:'DM Sans',sans-serif}.stApp h1,.stApp h2,.stApp h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.04em;color:#e9edf5}.stApp h1{font-size:2.15rem;margin:.12rem 0 1.4rem}.stApp h2{font-size:1.3rem;margin:1.7rem 0 .75rem}.stApp h3{font-size:1rem}.stSidebar{background:#10131a;border-right:1px solid #252a36}.stSidebar [data-testid="stRadio"] label{border-radius:8px;padding:8px 10px;margin:2px 0;transition:background .15s,color .15s}.stSidebar [data-testid="stRadio"] label:hover{background:#1a202b}.stSidebar [data-testid="stRadio"] label:has(input:checked){background:#202a3b;color:#f4f7fb}.stSidebar [data-testid="stRadio"] label:has(input:checked) p{font-weight:700}.stButton>button{background:#edf2fb!important;color:#11151d!important;border:0!important;border-radius:8px!important;font-weight:700!important;min-height:39px;padding:0 16px!important}.stButton>button:hover{background:#b9ccff!important;transform:translateY(-1px)}.metric-card{background:#121720;border:1px solid #293243;border-radius:10px;padding:17px 18px;min-height:98px;box-shadow:0 8px 20px rgba(0,0,0,.13)}.metric-label{color:#97a3b8;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em}.metric-value{font-family:'Space Grotesk';font-size:1.55rem;font-weight:700;margin-top:8px;color:#e9edf5}.panel{background:#111720;border:1px solid #293243;border-radius:10px;padding:18px 20px;margin:10px 0}.muted{color:#929db0}.good{color:#71d29a}.warn{color:#f0c35b}.stAlert{border-radius:9px}.stDataFrame{border:1px solid #293243;border-radius:10px;overflow:hidden}.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid #293243}.stTabs [data-baseweb="tab"]{height:38px;background:transparent;border-radius:7px 7px 0 0}.stTabs [aria-selected="true"]{background:#1c2636!important;color:#f2f6fd!important}.stSelectbox [data-baseweb="select"]>div,.stTextInput input,.stFileUploader{border-radius:8px!important;border-color:#30394a!important;background:#111720!important}.topbar{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #222a37;padding:0 0 14px;margin-bottom:18px}.eyebrow{font-size:.7rem;letter-spacing:.12em;color:#9ca9bd;font-weight:700}.status-pills{display:flex;gap:8px}.pill{font-size:.72rem;border:1px solid #2c3849;border-radius:999px;padding:5px 9px;color:#b9c6d9;background:#111720}.pill b{color:#72d29a;margin-right:4px}</style>''',unsafe_allow_html=True)
def card(label,value,detail=''):
 st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="muted">{detail}</div></div>',unsafe_allow_html=True)
def extract(result):
 rounds=result.get('rounds',[]); last=rounds[-1] if rounds else {}; return rounds,last,last.get('held_out_test',{}),last.get('run_configuration',{})
def safe_extract(upload):
 target=UPLOADS/f'dataset_{datetime.now():%Y%m%d_%H%M%S}';target.mkdir(parents=True)
 with zipfile.ZipFile(upload) as z:
  for m in z.infolist():
   if not str((target/m.filename).resolve()).startswith(str(target.resolve())):raise ValueError('Unsafe ZIP path')
  z.extractall(target)
 children=[p for p in target.iterdir() if p.is_dir()];return children[0] if len(children)==1 else target
def save_uploaded_images(label, uploads):
 """Save validated images locally under a class folder for future training."""
 from PIL import Image
 safe=re.sub(r'[^A-Za-z0-9_-]+','_',label).strip('_')
 if not safe: raise ValueError('Enter a class name using letters, numbers, hyphens, or underscores.')
 target=UPLOADS/'manual_dataset'/safe; target.mkdir(parents=True,exist_ok=True); saved=0
 for upload in uploads:
  suffix=Path(upload.name).suffix.lower()
  if suffix not in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}: raise ValueError(f'Unsupported file: {upload.name}')
  try: Image.open(upload).verify(); upload.seek(0)
  except Exception as exc: raise ValueError(f'Invalid image {upload.name}: {exc}')
  destination=target/f'{datetime.now():%Y%m%d_%H%M%S_%f}_{saved}{suffix}'
  destination.write_bytes(upload.getvalue()); saved+=1
 return target.parent,saved
def validate_dataset_folder(path):
 classes=[p for p in Path(path).iterdir() if p.is_dir()]
 count=sum(len([x for x in p.iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]) for p in classes)
 if len(classes)<2: raise ValueError('A classification dataset needs at least two class folders.')
 if count==0: raise ValueError('No supported images were found.')
 return classes,count
def start(cfg):
 label=f'web_{datetime.now():%Y%m%d_%H%M%S}';log=RESULTS/f'{label}.log';cmd=[sys.executable,'main.py','--rounds',str(cfg['rounds']),'--num-clients',str(cfg['clients']),'--local-epochs',str(cfg['epochs']),'--batch-size',str(cfg['batch']),'--label',label]
 if cfg['distribution']=='Non-IID':cmd+=['--non-iid']
 if cfg['data']:cmd+=['--data-dir',cfg['data']]
 f=open(log,'w',encoding='utf-8');p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));st.session_state.run={'p':p,'log':log,'label':label,'start':time.time()}
def run_state():
 r=st.session_state.get('run');
 if r:r['code']=r['p'].poll()
 return r
def header(title,kicker):
 st.markdown(f'<div class="topbar"><div><div class="eyebrow">{kicker}</div><h1>{title}</h1></div><div class="status-pills"><span class="pill"><b>●</b>Local engine</span><span class="pill">Research mode</span></div></div>',unsafe_allow_html=True)
def analytics(result,full=False):
 rounds,last,test,cfg=extract(result)
 if cfg.get('dataset_type')=='synthetic':st.warning('Synthetic dataset — metrics validate the software pipeline only, not medical or clinical performance.')
 cols=st.columns(5)
 for c,k,n in zip(cols,['accuracy','precision_macro','recall_macro','f1_macro','f1_weighted'],['Global accuracy','Precision','Recall','Macro F1','Weighted F1']):
  with c: card(n,f"{test.get(k,0):.3f}",'held-out test')
 if rounds:
  st.subheader('Learning trajectory');st.line_chart({'Global accuracy':[r['global_accuracy'] for r in rounds],'Global loss':[r['global_loss'] for r in rounds]})
  if full:
   st.subheader('Round metrics');st.dataframe([{'Round':r['round'],'Accuracy':r['global_accuracy'],'Loss':r['global_loss'],'Communication bytes':r['communication_bytes'],'Time (s)':r['round_seconds']} for r in rounds],width='stretch')
   st.subheader('Confusion matrix');st.dataframe(test.get('confusion_matrix',[]),width='stretch')
def overview(result):
 header('Research operations','OVERVIEW / FEDERATED SNN PLATFORM'); model=SpikingCNN(); rounds,last,test,cfg=extract(result or {})
 cols=st.columns(4)
 for c,a in zip(cols,[('System health','Online','local runtime'),('Experiments',str(len(files())),'saved local runs'),('Global model',f'{parameter_count(model):,}','trainable parameters'),('Hospital clients',str(cfg.get('num_clients',3)),'latest experiment')]):
  with c:card(*a)
 st.subheader('Research workflow');st.markdown('<div class="panel">Hospital datasets remain isolated <b>→</b> local LIF-SNN optimisation <b>→</b> parameter updates only <b>→</b> sample-weighted FedAvg <b>→</b> global checkpoint distribution.</div>',unsafe_allow_html=True)
 if result: st.subheader('Latest experiment'); analytics(result)
def experiments():
 header('Experiment workspace','CREATE / RUN / MONITOR');r=run_state();
 a,b,c,d=st.columns(4);clients=a.slider('Hospitals',2,8,3);rounds=b.slider('Rounds',1,10,3);epochs=c.slider('Local epochs',1,5,1);batch=d.selectbox('Batch size',[4,8,16,32],index=1)
 distribution=st.radio('Data distribution',['IID','Non-IID'],horizontal=True);source=st.radio('Dataset source',['Synthetic dataset','Upload image ZIP'],horizontal=True);data=''
 if source=='Upload image ZIP':
  up=st.file_uploader('Class-folder ZIP',type='zip');
  if up and st.button('Validate and prepare dataset'):
   try:
    folder=safe_extract(up);classes,count=validate_dataset_folder(folder);st.session_state.data=str(folder);st.success(f'Dataset prepared: {count} images across {len(classes)} classes.')
   except Exception as e:st.error(str(e))
  data=st.session_state.get('data','')
 if r and r['code'] is None:
  st.info(f"Experiment running for {time.time()-r['start']:.1f}s. New experiments are locked to prevent duplicate jobs.");st.code(Path(r['log']).read_text(encoding='utf-8') if Path(r['log']).exists() else 'Starting…')
  if st.button('Stop experiment safely'):r['p'].terminate();st.warning('Stop requested; incomplete runs are not added as results.')
 elif st.button('Start actual federated experiment',type='primary'):
  start({'clients':clients,'rounds':rounds,'epochs':epochs,'batch':batch,'distribution':distribution,'data':data});st.rerun()
 if r and r['code']==0:
  path=RESULTS/f"{r['label']}_results.json";st.success('Experiment completed and saved.')
  if path.exists():analytics(load(path),True)
def federated(result):
 header('Federated learning monitor','LIVE ARCHITECTURE / AGGREGATION');rounds,last,_,_=extract(result or {})
 st.markdown('<div class="panel" style="text-align:center;font-size:1.05rem">Hospital clients <b>→</b> Local LIF-SNN training <b>→</b> Weight transmission <b>→</b> Central FedAvg <b>→</b> Global model redistribution</div>',unsafe_allow_html=True)
 if last:st.columns(4)[0].metric('Communication round',last['round']);st.columns(4)[1].metric('Participants',len(last['clients']));st.columns(4)[2].metric('Bytes transferred',f"{last['communication_bytes']:,}");st.columns(4)[3].metric('Model version',f"v{last['round']}")
 else:st.info('Run an experiment to populate monitored federated state.')
def hospitals(result):
 header('Hospital clients','INDEPENDENT LOCAL TRAINING');_,last,_,_=extract(result or {})
 if not last:st.info('No client run available.');return
 for report,evaluation in zip(last['clients'],last['client_evaluations']):
  with st.expander(f"{report['client_id']}  •  completed",expanded=True):
   a,b,c,d=st.columns(4);a.metric('Local samples',evaluation['samples']);b.metric('Training loss',f"{report['train_loss']:.4f}");c.metric('Local accuracy',f"{evaluation['accuracy']:.3f}");d.metric('Model update',f"{report['update_bytes']:,} bytes")
def architecture():
 header('Model architecture','LIF SPIKING NEURAL NETWORK');m=SpikingCNN();st.markdown('<div class="panel"><b>Input</b> 1×28×28 → Conv 1→4 + LIF → MaxPool → Conv 4→8 + LIF → MaxPool → Dense 392→32 + LIF → Dense 32→4 + output LIF<br><br><span class="muted">Static images are temporally simulated for 10 steps. Leaky membrane dynamics and fast-sigmoid surrogate gradients enable BPTT training.</span></div>',unsafe_allow_html=True);st.metric('Trainable parameters',f'{parameter_count(m):,}');st.dataframe([{'Layer':'Conv2d','Shape':'1→4, kernel 3×3'},{'Layer':'Conv2d','Shape':'4→8, kernel 3×3'},{'Layer':'Linear','Shape':'392→32'},{'Layer':'Linear','Shape':'32→4'}],width='stretch')
def datasets():
 header('Dataset management','LOCAL DATA GOVERNANCE');st.info('Uploads stay on this laptop. The federated pipeline sends model parameters and aggregate metrics, never raw images. Use only authorized, de-identified data.')
 st.write('Preprocessing: grayscale conversion -> 28x28 resize -> tensor conversion -> normalization. Supported: JPG, JPEG, PNG, BMP, TIFF.')
 tab1,tab2=st.tabs(['Upload class-folder ZIP','Upload images by class'])
 with tab1:
  zip_upload=st.file_uploader('Upload a ZIP containing one folder per class',type='zip',key='dataset_zip')
  if zip_upload and st.button('Prepare ZIP dataset'):
   try:
    folder=safe_extract(zip_upload);classes,count=validate_dataset_folder(folder);st.session_state.data=str(folder);st.success(f'Ready for experiments: {count} images across {len(classes)} classes.')
   except Exception as exc:st.error(f'Dataset validation failed: {exc}')
 with tab2:
  label=st.text_input('Class label',placeholder='e.g., glioma')
  images=st.file_uploader('Choose one or more images for this class',type=['jpg','jpeg','png','bmp','tif','tiff'],accept_multiple_files=True,key='individual_images')
  if images and st.button('Add images to this class'):
   try:
    folder,count=save_uploaded_images(label,images);classes,total=validate_dataset_folder(folder);st.session_state.data=str(folder);st.success(f'Added {count} images. Dataset now has {total} images across {len(classes)} classes and is ready for experiments.')
   except Exception as exc:st.error(f'Could not add images: {exc}')
 st.caption('For direct image uploads, repeat “Add images to this class” for every class before starting training.')
 if UPLOADS.exists():
  rows=[]
  for p in UPLOADS.iterdir():
   if p.is_dir():
    try: classes,count=validate_dataset_folder(p);rows.append({'Dataset':p.name,'Classes':len(classes),'Images':count,'Status':'Ready locally'})
    except Exception: rows.append({'Dataset':p.name,'Classes':'Incomplete','Images':'-','Status':'Needs at least two classes'})
  if rows:st.subheader('Available local datasets');st.dataframe(rows,width='stretch')
def history():
 header('Experiment history','RESULTS / COMPARISON / EXPORT');fs=files()
 if not fs:st.info('No saved experiment records.');return
 p=st.selectbox('Saved experiment',fs,format_func=lambda x:x.name);r=load(p);analytics(r,True);st.download_button('Export report JSON',p.read_bytes(),file_name=p.name,mime='application/json')
def checkpoints():
 header('Model checkpoints','GLOBAL MODEL REGISTRY');
 if CHECKPOINT.exists():st.markdown(f'<div class="panel">Current global checkpoint <b>global_model.pt</b><br><span class="muted">{CHECKPOINT.stat().st_size/1024:.1f} KiB • CPU-compatible • produced by actual FedAvg aggregation</span></div>',unsafe_allow_html=True);st.download_button('Download checkpoint',CHECKPOINT.read_bytes(),'global_model.pt','application/octet-stream')
 else:st.info('No checkpoint available yet.')
def logs():
 header('System logs','LOCAL RUNTIME OBSERVABILITY');logs=sorted(RESULTS.glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)
 if logs: p=st.selectbox('Log file',logs,format_func=lambda x:x.name);st.code(p.read_text(encoding='utf-8'),language='text')
 else:st.info('No web experiment logs yet.')
def pi():
 header('Raspberry Pi deployment','EDGE CLIENT READINESS');st.warning('Physical Raspberry Pi status: not tested on hardware. The laptop simulation is separate from hardware validation.');st.code('python server.py --host 0.0.0.0 --port 8080 --rounds 3 --min_clients 1\n# On Raspberry Pi 3\nbash pi_setup/pi_install.sh ~/spikefed-iomt\nbash pi_setup/start_client.sh <SERVER_IP> 1 1.38 data',language='bash')
def main():
 st.set_page_config(page_title='Federated SNN Lab',page_icon='◈',layout='wide');css();
 if 'run' not in st.session_state:st.session_state.run=None
 with st.sidebar:
  st.markdown('## ◈ SNN Lab');st.markdown('<span class="good">●</span> <span class="muted">LOCAL SYSTEM ONLINE</span>',unsafe_allow_html=True);st.divider();page=st.radio('Navigation',NAV,label_visibility='collapsed');st.divider();st.caption('v1.0 • local-first research control plane')
 result=latest()
 {'Overview':lambda:overview(result),'Experiments':experiments,'Federated Learning':lambda:federated(result),'Hospital Clients':lambda:hospitals(result),'Model Architecture':architecture,'Dataset Management':datasets,'Performance Analytics':lambda:analytics(result,True) if result else st.info('Run an experiment first.'),'Experiment History':history,'Model Checkpoints':checkpoints,'System Logs':logs,'Raspberry Pi Deployment':pi,'Settings':lambda:st.info('Local, offline-first defaults are active. Device selection follows PyTorch CPU fallback.') }[page]()
if __name__=='__main__':main()
