"""Export task-specific tables and a test-only B preview without altering Fig5."""
from pathlib import Path
import json,shutil,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=Path(__file__).resolve().parent
P=['#274753','#297270','#299d8f','#8ab07c','#e7c66b','#f3a361','#e66d50']

def main():
    original=OUT/'original_fig5_snapshot';original.mkdir(exist_ok=True)
    for name in ['fig5.png','fig5.svg','fig5.ipynb','captions.md','draw_fig5.py']:
        if not (original/name).exists():shutil.copy2(OUT.parent/name,original/name)
    metrics=pd.read_csv(OUT/'module_metrics.csv');core=pd.read_csv(OUT/'same_core_metrics.csv');comp=pd.read_csv(OUT/'module_comparison.csv')
    for task in ['atpase_domains','kinase_domains']:
        for split in ['validation','test']:
            metrics[(metrics.task==task)&(metrics.partition=='new')&(metrics.split==split)].to_csv(OUT/f'{task}_{split}_metrics.csv',index=False)
    for split in ['validation','test']:
        core[(core.partition=='new')&(core.split==split)].to_csv(OUT/f'same_core_{split}_metrics.csv',index=False)
    summaries=pd.concat([pd.read_csv(OUT/f'{name}_split_summary.csv') for name in ['atpase','kinase']],ignore_index=True)
    summaries.to_csv(OUT/'split_summary.csv',index=False)
    selected=comp[(comp.partition=='new')&(comp.split=='test')&(comp.metric=='auroc')&(comp.combination_method=='combination_mean')]
    b=selected[(selected.task=='atpase_domains')|((selected.group.isin(['gly_loop_module','vaik_module']))&(selected.label.isin(['gly_loop_proxy_window3','VAIK_like_window3'])))].copy()
    b.to_csv(OUT/'fig5b_new_test_source.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(5.8,5.6),gridspec_kw={'height_ratios':[3,2]})
    for ax,task,groups,title in zip(axes,['atpase_domains','kinase_domains'],[['c_core','c_context','walker_b'],['gly_loop_module','vaik_module']],['ATPase · new test','Kinase · new test']):
        rows=pd.concat([b[(b.task==task)&(b.group==g)].sort_values('family') for g in groups]);y=np.arange(len(rows))
        names={'c_core':'Walker-A','c_context':'Context','walker_b':'Walker-B','gly_loop_module':'Gly-loop ±3','vaik_module':'VAIK ±3'}
        ax.hlines(y,rows.best_single_value,rows.combination_value,color='#bdc7c1',lw=2)
        ax.scatter(rows.best_single_value,y,color=P[0],s=25);ax.scatter(rows.combination_value,y,color=P[2],marker='D',s=25)
        ax.set_yticks(y,[f'{names[r.group]} · {r.family}' for r in rows.itertuples()]);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.set_title(title,loc='left')
    axes[-1].set_xlabel('AUROC')
    fig.legend(handles=[Line2D([],[],marker=m,ls='',color=c,label=l) for m,c,l in [('o',P[0],'Best group single on new test'),('D',P[2],'Fixed-group mean')]],loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.065,1,1],h_pad=2)
    for ext in ['png','svg']:fig.savefig(OUT/f'fig5b_new_test_preview.{ext}',dpi=300)
    plt.close(fig)
    clean=comp[(comp.partition=='new_excluding_train_duplicates')&(comp.split=='test')&(comp.metric=='auroc')&(comp.combination_method=='combination_mean')]
    sensitivity=b.merge(clean,on=['task','family','split','group','label','metric','combination_method'],suffixes=('_all','_clean'))
    sensitivity['combination_difference_after_exclusion']=sensitivity.combination_value_clean-sensitivity.combination_value_all
    sensitivity.to_csv(OUT/'fig5b_train_duplicate_sensitivity.csv',index=False)
    old=pd.read_csv(OUT.parent/'source_data/module_holdout_comparison.csv')
    old['task']=np.where(old.family.isin(['PF00004','PF00005']),'atpase_domains','kinase_domains')
    keys=['task','family','group','metric','combination_method']
    delta=b.merge(old,on=keys,suffixes=('_new','_old'))
    delta['combination_delta']=delta.combination_value_new-delta.combination_value_old
    delta.to_csv(OUT/'fig5b_old_vs_new.csv',index=False)
    def md_table(df):
        cols=list(df.columns);out=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
        for row in df.itertuples(index=False,name=None):out.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(out)
    report='''# Fig5 valid/test 重划分结果

原始 train/validation/test、历史指标和当前 Fig5 均未替换。原 Fig5 快照另存于 original_fig5_snapshot/。

## 划分

随机种子 20260910；将原 validation/test 合并，按已有 30% 同源 component（再合并完全相同序列关联的 component）整体随机分配，目标为每家族约 1:1 的序列数。只使用序列计数平衡，不查看标签或结果，也不搜索更好种子。每个新集合内部再按相同种子打散。

ATPase 不同 domain 与相同 C-core 两任务共用 atpase 划分；kinase 两家族共用 kinase 划分规则。不同任务和家族不混为一个训练任务。

'''+md_table(summaries)+'''

## B 的统一 new-test AUROC

该表沿用原 B 的定义：单项是当前评价 split 内该组 AUROC 最佳成员（描述性对照），组合为冻结归一化后的均值。所有行现在都来自 new test。

'''+md_table(b[['family','group','best_single','best_single_value','combination_value']])+'''

另提供 validation_selected_single_test_comparison.csv：仅用 new validation 选组内单项参考，再读取它在 new test 的值，供后续正式图选择。这不改变 latent 候选组、组合权重或归一化尺度。

## 保留原记录与重合敏感性

原 PF00004 test 的 6 条记录与 train 中一条相同序列重复；原 PF00069 test 有 1 条同类记录。按本次要求，它们保留在主要重划分及指标中；详情见 *_original_train_overlap.csv。新 valid 与 test 之间没有完全相同序列或现有 component 交叉，但不能宣称新集合与 train 完全序列隔离。new_excluding_train_duplicates 指标仅排除这些确切重合记录，不代表重建了完整同源聚类。

此操作是已使用 holdout 的重新划分，并非新增外部测试或全新盲测。它统一 B 的评价阶段，但不能消除先前查看结果的历史。

## 复现与文件

- repartition_and_evaluate.py：分组随机划分、CPU SAE 投影、训练尺度、全部新指标和旧指标复现。
- summarize_results.py：各任务导出、比较报告和 B 预览。
- splits/atpase 与 splits/kinase：新 validation/test CSV，保留 original_split、original_row 与同源组。
- module_metrics.csv / module_comparison.csv：原/new/排除重复敏感性全部指标。
- same_core_metrics.csv：共享与非共享 C-core latent 的新 AUROC/AUPRC、精确位置指标；位置矩阵另存 CSV。
- original_metric_reproduction.csv：计算实现对旧结果的复现误差。
- frozen_train_scales.csv：训练集 q99 尺度。ATPase 用同一训练数据和规则恢复；kinase 直接复用已保存尺度。
- activation_cache/：仅本次所需 SAE 选定列，可删除后复算，不包含新模型。
- fig5b_new_test_preview.png/svg：独立预览，未替换现有 Fig5。
'''
    (OUT/'结果说明.md').write_text(report)
    print(b[['family','group','best_single_value','combination_value']].to_string(index=False))
    print('REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
