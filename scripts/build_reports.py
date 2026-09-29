#!/usr/bin/env python3
"""Generate tables, plots and discussion drafts only from completed measured runs."""
import argparse
import csv
import html
import json
from pathlib import Path
import re
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from breastseg.core import grayscale_uint8,Gabor
from breastseg.external import load_external
from breastseg.statistics import bootstrap_ci,paired,holm

ROOT=Path(__file__).resolve().parents[1]
LABELS={'vanilla_b7':'Vanilla B7','gabor_b7':'Gabor bundle','gabor_boundary_b7':'Gabor + boundary',
 'scse_b7':'SCSE-only B7','residual_gabor_b7':'Residual Gabor','residual_sobel_b7':'Residual Sobel',
 'residual_gabor_boundary_b7':'Residual Gabor + boundary'}
METRICS=('dice','iou','boundary_f1_2px','surface_dice_2px','hd95_px')

def readcsv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def writecsv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze():
    out={};stats=[];summaries=[]
    for variant,label in LABELS.items():
        study='public_study' if variant in list(LABELS)[:3] else 'public_followup'
        ss=[json.loads((ROOT/'results'/study/f'{variant}_s{s}/summary.json').read_text()) for s in range(3)]
        rows=[sorted(readcsv(ROOT/'results'/study/f'{variant}_s{s}/test_per_image.csv'),key=lambda r:r['id']) for s in range(3)]
        assert all([r['id'] for r in rows[0]]==[r['id'] for r in rs] for rs in rows)
        vals={m:np.array([[float(r[m]) for r in rs] for rs in rows]) for m in METRICS}
        tr=[sorted(readcsv(ROOT/'results/tnbc_transfer'/f'{variant}_s{s}/per_image.csv'),key=lambda r:r['id']) for s in range(3)]
        groups=sorted({r['group'] for r in tr[0]})
        tv={m:np.array([[np.mean([float(r[m]) for r in rs if r['group']==g]) for g in groups] for rs in tr]) for m in METRICS}
        out[variant]=dict(label=label,study=study,runs=ss,rows=rows,values=vals,transfer=tv)
        record=dict(variant=variant,label=label,n_seeds=3,n_test_images=50,mean_val_dice=float(np.mean([s['best_val_dice'] for s in ss])),
            parameters=ss[0]['parameters'],seconds_mean=float(np.mean([s['seconds'] for s in ss])))
        for m,a in vals.items():
            ci=bootstrap_ci(a.mean(0));record.update({m:float(a.mean()),m+'_seed_sd':float(a.mean(1).std(ddof=1)),m+'_ci_low':ci[0],m+'_ci_high':ci[1]})
        record['tnbc_patient_dice']=float(tv['dice'].mean());record['tnbc_patient_dice_ci95']=bootstrap_ci(tv['dice'].mean(0))
        summaries.append(record)
    comparisons=[(v,'vanilla_b7') for v in LABELS if v!='vanilla_b7']+[
        ('gabor_boundary_b7','gabor_b7'),('residual_gabor_b7','scse_b7'),('residual_gabor_b7','residual_sobel_b7'),
        ('residual_gabor_boundary_b7','residual_gabor_b7')]
    for dataset,key in (('bbbc039','values'),('tnbc','transfer')):
        family=[]
        for a,b in comparisons:
            for m in ('dice','boundary_f1_2px'):
                p=paired(out[a][key][m].mean(0),out[b][key][m].mean(0))
                family.append(dict(dataset=dataset,a=a,b=b,metric=m,n_units=len(out[a][key][m].mean(0)),
                    mean_difference=p['mean_difference'],ci_low=p['ci95'][0],ci_high=p['ci95'][1],p=p['p']))
        for row,q in zip(family,holm([r['p'] for r in family])): row['p_holm']=q
        stats.extend(family)
    writecsv(ROOT/'results/study_summary.csv',summaries);writecsv(ROOT/'results/paired_statistics.csv',stats)
    (ROOT/'results/study_summary.json').write_text(json.dumps(summaries,indent=2))
    internal=json.loads((ROOT/'legacy/external_validation_20260928/outputs/internal_reference_summary.json').read_text())['summary']
    writecsv(ROOT/'results/previous_private_study.csv',[dict(condition=k,dice=v['dice'],dice_ci_low=v['dice_lo'],dice_ci_high=v['dice_hi'],n_seeds=v['n_seeds'],n_groups=v['n_groups']) for k,v in internal.items()])
    return out,summaries,stats

def figures(out,cases):
    dest=ROOT/'reports/figures';dest.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    for v,d in out.items():
        h=[json.loads((ROOT/'results'/d['study']/f'{v}_s{s}/history.json').read_text()) for s in range(3)]
        y=np.array([[r['val_dice'] for r in hh] for hh in h]);x=np.arange(1,21)
        axs[0].plot(x,y.mean(0),label=d['label']);axs[0].fill_between(x,y.mean(0)-y.std(0),y.mean(0)+y.std(0),alpha=.08)
    axs[0].set(xlabel='Epoch',ylabel='Full-validation mean Dice',ylim=(.4,1.0),title='All runs; shaded seed variation')
    axs[0].legend(fontsize=7,loc='lower right');axs[0].grid(alpha=.15)
    labels=[d['label'].replace(' + ',' +\n') for d in out.values()];x=np.arange(len(labels));w=.37
    for offset,m,label in ((-w/2,'dice','Dice'),(w/2,'boundary_f1_2px','Boundary F1 @ 2 px')):
        a=np.array([d['values'][m].mean(1) for d in out.values()])
        axs[1].bar(x+offset,a.mean(1)*100,w,yerr=a.std(1,ddof=1)*100,label=label,capsize=2)
    axs[1].set_xticks(x,labels,rotation=35,ha='right',fontsize=7);axs[1].set(ylabel='Test score (%)',ylim=(85,100),title='BBBC039 official test; mean ± seed SD')
    axs[1].legend(fontsize=8);axs[1].grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(dest/'learning_and_results.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.4,2.35))
    for offset,m,label in ((-w/2,'dice','Dice'),(w/2,'boundary_f1_2px','BF1 @ 2 px')):
        a=np.array([d['values'][m].mean(1) for d in out.values()])
        ax.bar(x+offset,a.mean(1)*100,w,yerr=a.std(1,ddof=1)*100,label=label,capsize=2)
    ax.set_xticks(x,['V','G','G+B','A','RG','RS','RG+B'],fontsize=8)
    ax.set_ylim(93,100);ax.set_ylabel('Test score (%)',fontsize=9);ax.tick_params(axis='y',labelsize=8)
    ax.legend(fontsize=8,loc='lower right');ax.grid(axis='y',alpha=.15);fig.tight_layout(pad=.5)
    fig.savefig(dest/'paper_results.pdf');fig.savefig(dest/'paper_results.png',dpi=220);plt.close(fig)
    # Representative cases are rank-selected, not manually picked for appearance.
    v='residual_gabor_b7';rows=sorted(out[v]['rows'][0],key=lambda r:float(r['dice']))
    chosen=[rows[int(round(q*(len(rows)-1)))] for q in (.1,.5,.9)]
    fig,axs=plt.subplots(3,4,figsize=(11,7.5));selected=[]
    for j,r in enumerate(chosen):
        c=cases['bbbc039'][r['id']];g=c['mask']>0;im=grayscale_uint8(c['image'])
        pred=cv2.imread(str(ROOT/'results/public_followup/residual_gabor_b7_s0/predictions'/f"{r['id']}.png"),0)>0
        axs[j,0].imshow(im,cmap='gray');axs[j,1].imshow(g,cmap='gray');axs[j,2].imshow(pred,cmap='gray')
        error=np.repeat((im/255*.55)[...,None],3,2);error[pred&~g]=[1,.2,.1];error[~pred&g]=[.15,.5,1]
        axs[j,3].imshow(error)
        axs[j,0].set_ylabel(f"q{[10,50,90][j]}\nDice {float(r['dice']):.4f}",fontsize=9)
        for ax in axs[j]:ax.set_xticks([]);ax.set_yticks([])
        selected.append(dict(quantile=[.1,.5,.9][j],id=r['id'],dice=float(r['dice'])))
    for ax,t in zip(axs[0],('Native fluorescence','Reference foreground','Residual Gabor prediction','Red: FP; blue: FN')):ax.set_title(t,fontsize=9)
    fig.tight_layout();fig.savefig(dest/'bbbc039_examples.png',dpi=160);plt.close(fig)
    (dest/'example_selection.json').write_text(json.dumps(selected,indent=2))
    c=cases['bbbc039'][chosen[1]['id']];im=grayscale_uint8(c['image']);f=im.astype(np.float32)/255
    sobel=cv2.magnitude(cv2.Sobel(f,cv2.CV_32F,1,0),cv2.Sobel(f,cv2.CV_32F,0,1))
    fig,axs=plt.subplots(1,4,figsize=(11,3))
    for ax,img,title in zip(axs,(im,Gabor()(im),sobel,c['mask']),('Actual input','Gabor response (24 filters)','Sobel magnitude','Reference foreground')):
        ax.imshow(img,cmap='gray');ax.set_title(title,fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(dest/'actual_edge_features.png',dpi=180);plt.close(fig)
    tr=readcsv(ROOT/'results/tnbc_transfer/residual_gabor_b7_s0/per_image.csv');tr.sort(key=lambda r:float(r['dice']));r=tr[len(tr)//2];c=cases['tnbc'][r['id']]
    pred=cv2.imread(str(ROOT/'results/tnbc_transfer/residual_gabor_b7_s0/predictions'/f"{r['id']}.png"),0)>0
    fig,axs=plt.subplots(1,3,figsize=(9,3));axs[0].imshow(c['image']);axs[1].imshow(c['mask'],cmap='gray');axs[2].imshow(pred,cmap='gray')
    for ax,t in zip(axs,(f"TNBC {r['id']} (median-ranked)",'Reference nuclei',f"Transfer prediction: Dice {float(r['dice']):.3f}")):ax.set_title(t,fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(dest/'tnbc_transfer_example.png',dpi=180);plt.close(fig)

def render_pdf(path):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    for name,file in (('DejaVu','DejaVuSans.ttf'),('DejaVu-Bold','DejaVuSans-Bold.ttf')):pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold')
    styles=getSampleStyleSheet()
    for name in ('Normal','Title','Heading1','Heading2'):
        styles[name].fontName='DejaVu' if name=='Normal' else 'DejaVu-Bold'
    styles['Normal'].fontSize=9;styles['Normal'].leading=12;styles['Normal'].spaceAfter=6
    styles['Title'].fontSize=17;styles['Title'].leading=21
    styles['Heading1'].fontSize=12;styles['Heading1'].leading=16
    styles['Heading2'].fontSize=10;styles['Heading2'].leading=14
    cell=ParagraphStyle('Cell',parent=styles['Normal'],fontSize=7,leading=9,spaceAfter=0)
    def inline(s):
        s=html.escape(s);s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s);s=s.replace('`','')
        return re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#176b80">\1</link>',s)
    width=7.1*inch;doc=SimpleDocTemplate(str(path.with_suffix('.pdf')),pagesize=(8.5*inch,11*inch),leftMargin=.7*inch,rightMargin=.7*inch,topMargin=.6*inch,bottomMargin=.6*inch)
    story=[];lines=path.read_text().splitlines();i=0
    while i<len(lines):
        s=lines[i].strip()
        if not s:i+=1;continue
        if s.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                values=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[-: ]+',v) for v in values):rows.append([Paragraph(inline(v),cell) for v in values])
                i+=1
            n=len(rows[0]);widths=[width/n]*n
            if n>=5:widths=[width*.25]+[width*.75/(n-1)]*(n-1)
            t=Table(rows,colWidths=widths,repeatRows=1)
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf2')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8fa')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([t,Spacer(1,10)]);continue
        im=re.match(r'!\[.*?\]\((.*?)\)',s)
        if im:
            image=Image(str((path.parent/im.group(1)).resolve()));ratio=min(width/image.imageWidth,5.1*inch/image.imageHeight)
            image.drawWidth=image.imageWidth*ratio;image.drawHeight=image.imageHeight*ratio;story.extend([image,Spacer(1,8)])
        elif s.startswith('# '):story.append(Paragraph(inline(s[2:]),styles['Title']))
        elif s.startswith('## '):story.append(Paragraph(inline(s[3:]),styles['Heading1']))
        elif s.startswith('### '):story.append(Paragraph(inline(s[4:]),styles['Heading2']))
        else:
            paragraph=[s]
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','|','![')):
                i+=1;paragraph.append(lines[i].strip())
            story.append(Paragraph(inline(' '.join(paragraph)),styles['Normal']))
        i+=1
    def footer(c,d):
        c.setFont('DejaVu',7);c.drawString(.7*inch,.3*inch,'Research discussion draft | measured results | 29 September 2026');c.drawRightString(7.8*inch,.3*inch,str(d.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
    out,ss,stats=analyze();cases={ds:{c['id']:c for c in load_external(a.data,ds)} for ds in ('bbbc039','tnbc')};figures(out,cases)
    by={r['variant']:r for r in ss};best_val=max(ss,key=lambda r:r['mean_val_dice'])
    public='| Model | Dice % ± seed SD | IoU % | BF1@2px % | HD95 px |\n|---|---|---|---|---|\n'
    transfer='| Model | TNBC patient-macro Dice % | Descriptive 95% CI % |\n|---|---|---|\n'
    texrows=[]
    for r in ss:
        public+=f"| {r['label']} | {r['dice']*100:.2f} ± {r['dice_seed_sd']*100:.2f} | {r['iou']*100:.2f} | {r['boundary_f1_2px']*100:.2f} | {r['hd95_px']:.2f} |\n"
        lo,hi=r['tnbc_patient_dice_ci95'];transfer+=f"| {r['label']} | {r['tnbc_patient_dice']*100:.2f} | {lo*100:.2f}–{hi*100:.2f} |\n"
        texrows.append(f"{r['label'].replace(' + ',' + ')} & {100*r['dice']:.2f} $\\pm$ {100*r['dice_seed_sd']:.2f} & {100*r['boundary_f1_2px']:.2f} & {100*r['tnbc_patient_dice']:.2f} \\\\")
    prior={ds:json.loads((ROOT/'legacy/external_validation_20260928/outputs'/f'{ds}_zero_shot/summary.json').read_text()) for ds in ('bbbc038','bbbc039','tnbc')}
    frozen='| Dataset | Released 2D Dice % | Revised 2D Dice % | Otsu Dice % | Cellpose-SAM Dice % |\n|---|---|---|---|---|\n'
    for ds,d in prior.items():frozen+=f"| {ds.upper()} | "+' | '.join(f"{d[k]['dice']*100:.2f}" for k in ('released_2d','revised_2d_s0','otsu_border_polarity','cellpose_sam'))+' |\n'
    legacy=[]
    for path in sorted((ROOT/'results/legacy_external').glob('*/*/summary.json')):
        s=json.loads(path.read_text());legacy.append(dict(dataset=s['dataset'],version=s['source_version'],experiment=s['experiment'],dice=s['mean']['dice'],boundary_f1_2px=s['mean']['boundary_f1_2px'],empty_predictions=int(s['mean']['empty_prediction']*s['n']),checkpoint_sha256=s['checkpoint_sha256']))
    if len(legacy)!=72:raise RuntimeError(f'Expected 72 historical checkpoint/dataset evaluations, found {len(legacy)}')
    writecsv(ROOT/'results/legacy_external_summary.csv',legacy)
    mech=[]
    for r in stats:
        if r['dataset']=='bbbc039' and r['metric']=='dice' and (r['a'],r['b']) in [('gabor_b7','vanilla_b7'),('gabor_boundary_b7','gabor_b7'),('residual_gabor_b7','scse_b7'),('residual_gabor_b7','residual_sobel_b7'),('residual_gabor_boundary_b7','residual_gabor_b7')]:
            mech.append(f"{LABELS[r['a']]} versus {LABELS[r['b']]}: Dice difference {r['mean_difference']*100:+.3f} percentage points, paired image-bootstrap interval [{r['ci_low']*100:+.3f}, {r['ci_high']*100:+.3f}], Holm-adjusted p={r['p_holm']:.4g}.")
    vals=dict(PUBLIC_TABLE=public.strip(),TRANSFER_TABLE=transfer.strip(),FROZEN_TABLE=frozen.strip(),MECHANISTIC='\n\n'.join(mech),
        VANILLA_DICE=f"{100*by['vanilla_b7']['dice']:.2f}",GABOR_DICE=f"{100*by['gabor_b7']['dice']:.2f}",BOUNDARY_DICE=f"{100*by['gabor_boundary_b7']['dice']:.2f}",
        RESIDUAL_DICE=f"{100*by['residual_gabor_b7']['dice']:.2f}",RESIDUAL_TNBC=f"{100*by['residual_gabor_b7']['tnbc_patient_dice']:.2f}",
        GABOR_TNBC=f"{100*by['gabor_b7']['tnbc_patient_dice']:.2f}",
        BEST_VAL_MODEL=best_val['label'],BEST_VAL_DICE=f"{100*best_val['dice']:.2f}",TEX_ROWS='\n'.join(texrows),
        TOTAL_RUNS='21',TOTAL_EPOCHS='420',LEGACY_EVALUATIONS=str(len(legacy)),
        MEAN_GAIN=f"{100*(by['residual_gabor_b7']['dice']-by['gabor_b7']['dice']):+.2f}",
        VS_VANILLA=f"{100*(by['residual_gabor_b7']['dice']-by['vanilla_b7']['dice']):+.2f}")
    primary=next(r for r in stats if r['dataset']=='bbbc039' and r['a']=='gabor_b7' and r['b']=='vanilla_b7' and r['metric']=='dice')
    mantissa,exponent=f"{primary['p_holm']:.2e}".split('e')
    vals.update(GABOR_DELTA=f"{primary['mean_difference']*100:.3f}",GABOR_CI_LOW=f"{primary['ci_low']*100:.3f}",
        GABOR_CI_HIGH=f"{primary['ci_high']*100:.3f}",GABOR_P_TEX=mantissa+'\\times10^{'+str(int(exponent))+'}')
    (ROOT/'results/report_values.json').write_text(json.dumps(vals,indent=2))
    for template in (ROOT/'reports/meeting_report.md.in',ROOT/'paper/first_draft.md.in',ROOT/'paper/first_draft_isbi.tex.in'):
        text=template.read_text()
        for key,value in vals.items():text=text.replace('{{'+key+'}}',value)
        if re.search(r'\{\{[A-Z_]+\}\}',text):raise ValueError(f'Unfilled template: {template}')
        path=template.with_suffix('');path.write_text(text)
        if path.suffix=='.md':render_pdf(path)
    for name in ('code_audit','reviewer_action_matrix','next_steps'):render_pdf(ROOT/'reports'/f'{name}.md')
    print(json.dumps({k:v for k,v in vals.items() if 'TABLE' not in k and k not in ('TEX_ROWS','MECHANISTIC')},indent=2))

if __name__=='__main__':main()
