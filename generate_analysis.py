"""Create evidence-based paper-aligned analysis from saved IID/non-IID records."""
from pathlib import Path
import csv,json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image

ROOT=Path(__file__).parent; OUT=ROOT/'experimental_analysis'; GR=OUT/'research_graphs'; CM=OUT/'confusion_matrices'
OUT.mkdir(exist_ok=True); GR.mkdir(exist_ok=True); CM.mkdir(exist_ok=True)
iid=json.loads((ROOT/'results/iid_results.json').read_text()); non=json.loads((ROOT/'results/non_iid_results.json').read_text())
def last(x):return x['rounds'][-1]
def f(x):return last(x).get('held_out_test',{})
def write_csv(name,heads,rows):
 with open(OUT/name,'w',newline='',encoding='utf-8') as h:csv.writer(h).writerows([heads,*rows])
def p(x):return f'{x:.4f}' if isinstance(x,float) else str(x)

arch=[['Component','Actual implementation / value','Evidence category'],['Input','1 x 28 x 28 grayscale image','Source code'],['Conv1 + LIF','Conv2d 1->4, 3x3 padding 1; Leaky LIF','Source code'],['Pool1','MaxPool2d 2','Source code'],['Conv2 + LIF','Conv2d 4->8, 3x3 padding 1; Leaky LIF','Source code'],['Pool2','MaxPool2d 2; flattened 8 x 7 x 7 = 392','Source code'],['Dense + LIF','Linear 392->32; Leaky LIF','Source code'],['Output + LIF','Linear 32->4; Leaky LIF','Source code'],['Temporal steps','10','Source code'],['Membrane decay beta','0.9','Source code'],['Threshold theta','Library default (not explicitly configured)','Source audit'],['Surrogate gradient','snnTorch fast_sigmoid(slope=25)','Source code'],['Encoding','Static normalized image presented each time step','Source code'],['Reset mechanism','snnTorch Leaky library default; not explicitly configured','Source audit'],['Trainable parameters','13,044','Measured/source'],['Checkpoint size',f"{(ROOT/'checkpoints/global_model.pt').stat().st_size} bytes",'Measured filesystem']]
write_csv('architecture_table.csv',arch[0],arch[1:])
settings=[['Setting','Actual value','Evidence category'],['Dataset','SyntheticMedicalImages (pipeline-test only)','Source/results'],['Total images','96 (24 per synthetic class x 4)','Source code'],['Split','70% train = 67; 15% validation = 14; 15% test = 15','Source code/calculated'],['Classes','4 synthetic classes','Source code'],['Image processing','grayscale; resize 28x28; ToTensor; Normalize(mean=0.5,std=0.5)','Source code'],['Clients','3','Saved IID/non-IID records'],['Rounds','3','Saved IID/non-IID records'],['Local epochs','1','main.py default/config'],['Batch size','8','Saved run configuration/default'],['Optimizer','Adam','Source code'],['Learning rate','0.001','Source code'],['Loss','snnTorch ce_rate_loss','Source code'],['Seed','42','main.py default'],['Device','CPU in saved runs','Run output/config'],['Precision','PyTorch default float32; not explicitly logged','Source audit'],['FedAvg','sample-count weighted mean','Source code']]
write_csv('experimental_settings.csv',settings[0],settings[1:])
rows=[]
for name,x in [('IID',iid),('Non-IID',non)]:
 l=last(x);t=f(x);rows.append([name,3,l['global_accuracy'],t['accuracy'],t['precision_macro'],t['recall_macro'],t['f1_macro'],t['f1_weighted'],l['global_loss'],t['loss'],sum(r['round_seconds'] for r in x['rounds'])])
write_csv('results_comparison.csv',['Configuration','Rounds','Final local-evaluation global accuracy','Held-out accuracy','Macro precision','Macro recall','Macro F1','Weighted F1','Final global loss','Held-out loss','Measured round-time sum (s)'],rows)
comm=[]
for name,x in [('IID',iid),('Non-IID',non)]:
 per=x['rounds'][0]['communication_bytes']; rounds=len(x['rounds']); upload=per*rounds;download=per*rounds;comm.append([name,per,52256,rounds,upload,download,upload+download,'Download calculated assuming global state distributed to all 3 participants; no transport framing/TLS measured'])
write_csv('communication_analysis.csv',['Configuration','Measured upload bytes/round','Measured update bytes/client','Rounds','Calculated total upload bytes','Calculated total download bytes','Calculated parameter-exchange bytes','Scope'],comm)

def plot_curves():
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for name,x,color in [('IID',iid,'#247ba0'),('Non-IID',non,'#e76f51')]:
  r=x['rounds'];axs[0].plot([q['round'] for q in r],[q['global_accuracy'] for q in r],marker='o',label=name,color=color);axs[1].plot([q['round'] for q in r],[q['global_loss'] for q in r],marker='o',label=name,color=color)
 for ax,title,y in zip(axs,['Global local-evaluation accuracy','Global loss'],['Accuracy','Rate loss']):ax.set(title=title,xlabel='Communication round',ylabel=y);ax.grid(alpha=.25);ax.legend()
 fig.tight_layout();fig.savefig(GR/'iid_vs_non_iid_curves.png',dpi=180);plt.close(fig)
def plot_clients():
 fig,ax=plt.subplots(figsize=(7,4));names=[q['client_id'] for q in last(non)['client_evaluations']];vals=[q['accuracy'] for q in last(non)['client_evaluations']];ax.bar(names,vals,color='#247ba0');ax.set(ylim=(0,1),title='Non-IID client-wise accuracy (round 3)',ylabel='Accuracy');fig.tight_layout();fig.savefig(GR/'non_iid_client_accuracy.png',dpi=180);plt.close(fig)
def plot_comm():
 fig,ax=plt.subplots(figsize=(6,4));ax.bar(['IID','Non-IID'],[sum(q['communication_bytes'] for q in iid['rounds']),sum(q['communication_bytes'] for q in non['rounds'])],color=['#247ba0','#e76f51']);ax.set(title='Measured client upload bytes across 3 rounds',ylabel='Bytes');fig.tight_layout();fig.savefig(GR/'communication_upload_bytes.png',dpi=180);plt.close(fig)
def plot_cm(name,x):
 matrix=f(x)['confusion_matrix'];fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(matrix,cmap='Blues');fig.colorbar(im,ax=ax);ax.set(title=f'{name} held-out synthetic confusion matrix',xlabel='Predicted class',ylabel='Actual class');
 for i,row in enumerate(matrix):
  for j,v in enumerate(row):ax.text(j,i,str(v),ha='center',va='center')
 fig.tight_layout();fig.savefig(CM/f'{name.lower().replace("-","_")}_confusion_matrix.png',dpi=180);plt.close(fig)
plot_curves();plot_clients();plot_comm();plot_cm('IID',iid);plot_cm('Non-IID',non)
# Simple source-based workflow/architecture figures.
for filename,text in [('federated_workflow.png','Hospital 1/2/3\nlocal LIF-SNN\n      ↓\nmodel weights only\n      ↓\nweighted FedAvg\n      ↓\nglobal state redistributed'),('snn_architecture.png','1×28×28 Input → Conv(1→4)+LIF → Pool → Conv(4→8)+LIF → Pool → Dense(392→32)+LIF → Dense(32→4)+LIF')]:
 fig,ax=plt.subplots(figsize=(10,2.5));ax.axis('off');ax.text(.5,.5,text,ha='center',va='center',fontsize=14,bbox=dict(boxstyle='round,pad=.8',fc='#edf4fa',ec='#247ba0'));fig.tight_layout();fig.savefig(GR/filename,dpi=180);plt.close(fig)

check='''# Missing Experiments and Measurement Checklist

## Completed (synthetic laptop simulation)
- [x] Genuine LIF-SNN source implementation and 13,044-parameter count.
- [x] Three-client IID and Non-IID FedAvg simulations, three rounds each.
- [x] Held-out synthetic metrics, per-client metrics, confusion matrices, update-byte measurements, and per-round timing.
- [x] Checkpoint saved and reloaded in automated tests.

## Partially completed
- [~] Communication: tensor upload bytes are measured. Downlink is calculated under an explicit model-distribution assumption; protocol framing, TLS, and retries are not measured.
- [~] Dataset isolation is implemented in the simulation loader. Real patient-level split validation has not been performed.

## Missing / NOT MEASURED
- [ ] Central CNN baseline.
- [ ] Central SNN baseline.
- [ ] Federated CNN baseline.
- [ ] Real medical dataset and clinical validation.
- [ ] Raspberry Pi 3 inference/training time, CPU, RAM, temperature, power, and network measurements.
- [ ] CPU utilization, peak RAM, inference latency, aggregation-only time, and validation-loss tracking.
- [ ] Cryptographic secure aggregation, differential privacy, and formal privacy guarantees.

No physical Raspberry Pi result is claimed in this report.
'''
(OUT/'missing_experiments_checklist.md').write_text(check,encoding='utf-8')
def md_table(rows):
 return '\n'.join(['| '+' | '.join(map(str,r))+' |' for r in rows])
report=f'''# Complete Experimental Report

## Scope and evidence
This report extracts source-defined and saved-run information from the local project. No retraining was performed. The conference paper itself was not present in the supplied workspace; Tables I-V follow the requested structure. All evaluated outcomes below are **synthetic pipeline-validation results**, not medical diagnostic or clinical validation.

## Table I - Actual SNN Architecture
{md_table(arch)}

## Table II - Expected Effects of Parameter Changes
| Parameter change | Theoretical expectation | Observed evidence in this project |
| --- | --- | --- |
| More temporal steps | Greater temporal compute and memory; may change spike-rate decisions | Not experimentally varied. |
| Higher beta | Longer membrane retention | Not experimentally varied; beta fixed at 0.9. |
| More local epochs | More local computation and possible client drift | Not experimentally varied; one epoch used. |
| More clients/rounds | More parameter traffic and computation | Three clients/three rounds measured only. |
| Non-IID partition | Distribution skew can impede convergence | Round-3 global local-eval accuracy: IID {last(iid)['global_accuracy']:.4f}; Non-IID {last(non)['global_accuracy']:.4f}. |

## Table III - Client Deployment Configurations
| Configuration | Status | Evidence |
| --- | --- | --- |
| Laptop simulation | Executed | CPU, three local simulated clients, synthetic data. |
| Hybrid server + Pi client | Prepared, not executed | Flower server/client and Pi scripts exist; no hardware measurements. |
| Physical Raspberry Pi deployment | Not measured | Hardware test unavailable. |

## Table IV - Actual Experimental Settings
{md_table(settings)}

## Table V - Actual Experimental Results
| Method | IID held-out accuracy | Non-IID held-out accuracy | Status |
| --- | --- | --- | --- |
| Federated SNN | {f(iid)['accuracy']:.4f} | {f(non)['accuracy']:.4f} | Executed synthetic runs |
| Central CNN | N/A | N/A | Not implemented/evaluated |
| Central SNN | N/A | N/A | Not implemented/evaluated |
| Federated CNN | N/A | N/A | Not implemented/evaluated |

## IID versus Non-IID comparison
IID final local-evaluation global accuracy was {last(iid)['global_accuracy']:.4f}; Non-IID was {last(non)['global_accuracy']:.4f}. Both held-out synthetic accuracy values were {f(iid)['accuracy']:.4f}. Measured sums of communication-round duration were {sum(q['round_seconds'] for q in iid['rounds']):.3f}s (IID) and {sum(q['round_seconds'] for q in non['rounds']):.3f}s (Non-IID). Each configuration recorded 156,768 measured upload bytes per round (three 52,256-byte state updates). The result files do not isolate aggregation time or validation loss, so those are NOT MEASURED.

## Communication calculation
FedAvg uses local sample counts as weights: W = sum(n_k W_k) / sum(n_k). For each three-round run, measured parameter uploads are 156,768 x 3 = 470,304 bytes. If the global state is redistributed to each of the three clients with the same tensor size, calculated parameter downloads are also 470,304 bytes; combined parameter exchange is 940,608 bytes. These calculations exclude network/protocol overhead.

## Computational performance
Checkpoint size is {(ROOT/'checkpoints/global_model.pt').stat().st_size} bytes. Per-round time and local training time are saved. CPU utilization, RAM/peak memory, inference latency, and Raspberry Pi metrics were not recorded and are explicitly NOT MEASURED.

## Required next experiments
See `missing_experiments_checklist.md` for the evidence-based gap list.
'''
(OUT/'complete_experimental_report.md').write_text(report,encoding='utf-8')
# PDF
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='Small',parent=styles['BodyText'],fontSize=8,leading=10));story=[Paragraph('Complete Experimental Report',styles['Title']),Paragraph('Evidence-based extraction from saved synthetic laptop simulations. Not clinical validation.',styles['BodyText']),Spacer(1,12)]
for title,data in [('Table I - Actual SNN Architecture',arch),('Table IV - Actual Experimental Settings',settings),('Table V - Results Comparison',[['Configuration','Global accuracy','Held-out accuracy','Macro F1','Round-time sum (s)'],*[ [r[0],p(r[2]),p(r[3]),p(r[6]),p(r[-1])] for r in rows]])]:
 story+=[Paragraph(title,styles['Heading2'])];t=Table(data,repeatRows=1,colWidths=[4*cm,10*cm,4*cm] if len(data[0])==3 else [3.2*cm]*len(data[0]));t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1f4e79')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.25,colors.grey),('FONTSIZE',(0,0),(-1,-1),7),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf3f8')]) ]));story+=[t,Spacer(1,12)]
story+=[Paragraph('Figures from actual saved results',styles['Heading1']),Image(str(GR/'iid_vs_non_iid_curves.png'),width=17*cm,height=6.8*cm),Spacer(1,8),Image(str(CM/'non_iid_confusion_matrix.png'),width=9*cm,height=7.2*cm),Paragraph('Missing measurements: CPU/RAM/peak memory, inference latency, aggregation-only time, baseline comparisons, and physical Raspberry Pi metrics.',styles['Small'])]
SimpleDocTemplate(str(OUT/'complete_experimental_report.pdf'),pagesize=A4,rightMargin=1.3*cm,leftMargin=1.3*cm,topMargin=1.3*cm,bottomMargin=1.3*cm).build(story)
print('Generated',OUT)
