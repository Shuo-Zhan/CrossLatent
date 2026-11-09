"""Assemble auditable result tables and a replayable notebook after inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import sys

OUT=Path(__file__).resolve().parent

def table(frame):
    def fmt(x):
        if isinstance(x,(float,np.floating)):
            return f'{x:.4f}' if np.isfinite(x) else 'NA'
        return str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])

def main():
    frame=pd.read_csv(OUT/'selected_sequences.csv',dtype={'taxid':str,'mask_target_domain':str})
    checks=json.loads((OUT/'evaluation_checks.json').read_text())
    selection=json.loads((OUT/'selection_manifest.json').read_text())
    audit=pd.read_csv(OUT/'candidate_audit.csv')
    homology=pd.read_csv(OUT/'homology_audit.csv')
    counts=json.loads((OUT/'eligible_cluster_counts.json').read_text())
    cohort=[]
    for fam,d in frame.groupby('family'):
        h=homology[homology.family==fam]
        cohort.append(dict(family=fam,downloaded=int((audit.family==fam).sum()),before_homology=len(h),after_homology=int((~h.excluded_near_old).sum()),eligible_cluster_representatives=counts[fam],selected=len(d),taxa=d.taxid.nunique(),tokens=int(d.length.sum()),median_length=float(d.length.median()),max_detected_old_identity=float(d.identity.max())))
    cohort=pd.DataFrame(cohort);cohort.to_csv(OUT/'cohort_summary.csv',index=False)
    main_results=pd.read_csv(OUT/'fig5b_external_metrics.csv')
    modules=pd.read_csv(OUT/'module_metrics.csv')
    ploop=pd.read_csv(OUT/'ploop_metrics.csv')
    shared=ploop[ploop.expected_offset.notna()]
    ann=pd.read_csv(OUT/'sequence_annotations.csv')
    summary=[]
    for fam,d in shared.groupby('family'):
        annotation=ann[ann.family==fam]
        summary.append(dict(family=fam,n_sequences=200,sequences_with_core=int((annotation.n_cores>0).sum()),n_cores=int(d.n_cores.iloc[0]),mean_core_auroc=float(d.core_auroc.mean()),mean_position_auprc=float(d.position_auprc.mean()),min_position_auprc=float(d.position_auprc.min()),mean_strict_within_core_top1=float(d.strict_within_core_top1.mean()),expected_matrix_peaks=int((d.expected_offset==d.mean_response_peak_offset).sum())))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'ploop_summary.csv',index=False)
    comparisons=pd.read_csv(OUT/'module_comparison_with_original_test.csv')
    comparisons['delta_auroc']=comparisons.auroc_external-comparisons.auroc_original
    comparisons['delta_auprc']=comparisons.auprc_external-comparisons.auprc_original
    comparisons.to_csv(OUT/'module_comparison_with_original_test.csv',index=False)
    coverage=pd.read_csv(OUT/'motif_coverage.csv')
    failure_count=checks['conditional_within_core_failed_pairs']
    report=f'''# CrossLatent：四家族各 200 条新序列验证

## 实验范围与数据独立性

共 800 条蛋白质：PF00004、PF00005、PF00069、PF07714 各 200 条。P-loop 实验完整复用 PF00004/PF00005 的同一批 400 条，未另按 motif 存在或预测表现选样。上一轮 64 条 pilot 已排除。下载采用 InterPro 公共 UniProt/Pfam API，原始分页、请求地址和 SHA256 保存在 raw/ 和 acquisition_*.json。

{table(cohort)}

参考排除池为旧本地 505,848 条唯一序列加 pilot 64 条。先排除相同序列和 accession，再排除 MMseqs2 检测到 identity ≥80%、双方覆盖均 ≥50% 的匹配；新候选按 identity ≥80%、双方覆盖均 ≥80% 聚类，取每簇一个代表，最后按固定哈希顺序进行物种轮转抽样。最初每家族下载 2,000 条，如不足 200 个代表则补下载。具体下载量以上表及原始记录为准。

本批是新获取、相对于参考池经过排除的数据；不等于最近收录、完全远缘、或未出现在 ESM2 预训练中。API 顺序候选池及物种构成可能带来抽样偏倚。聚类代表和 taxon 数均不等同于独立生物学重复数量。

## Fig. 5 B 对应模块：固定 mean，AUROC

沿用原模块成员、原家族训练集 q99 缩放及 [0,1] 截断，在目标结构域内部比较核心标签与其他残基。没有重新选择 latent、调整模块成员、训练模型或使用新集归一化。AUROC 为主指标；AUPRC 同时列出，其随机排序基线约为 prevalence。

{table(main_results[['family','group','label','auroc','auprc','prevalence']])}

## 所有预先固定模块（包括扩展催化模块）

{table(modules[modules.method=='combination_mean'][['family','group','label','auroc','auprc','prevalence']])}

单 latent 结果也保存在 module_metrics.csv；未依据新集挑选“最佳单 latent”。domain_shared_individual_metrics.csv 额外记录原 ATPase 18 个、kinase 63 个 shared latent 对结构域标签的逐一表现。

## P-loop：相同 400 条蛋白上的 8 个位置特征

{table(summary)}

上述均值是 8 个固定 latent 指标的算术均值，不是新训练或组合分类器的指标。position AUPRC 的负类包括整条蛋白其余残基；whole-core AUROC 的正类为全部 P-loop 核心残基。未解析到核心的蛋白仍保留在全序列背景中。

within-core top-1 只在已经由标签确定的 8 残基核心窗口内判断峰值位置。严格版本另外要求激活 >0 且最大值唯一。失败的核心×latent 组合数：{failure_count}，逐项见 ploop_within_core_failures.csv。该条件性指标不能解释为整条蛋白上的定位准确率。

## 标签覆盖与证据边界

{table(coverage[['family','label','n_sequences','n_detected','detection_fraction','n_positive']])}

ATPase 使用原域内最早 PS00017-like pattern、21 aa 周边窗口及 Walker-B 规则；kinase 使用原 gly-loop、VAIK-like、HRD、DFG 的相对位置规则与 ±3 aa 窗口。它们是序列定义的功能/结构代理标签。没有 motif 匹配不表示不存在真实结构，也没有直接验证三维几何或结构因果性。若某一模块在新集较弱，应如实限定该模块结论，不能由其他模块的强表现替代。

## 与现有 test 的比较

module_comparison_with_original_test.csv 和 ploop_comparison_with_original_test.csv 对照清理后的现有 Fig. 5 test（new_excluding_train_duplicates）。两个数据集的序列组成、长度和正类比例不同，数值升降属于描述性比较，不能直接称为模型改进。taxon_equal_module_intervals.csv 给出物种等权的、已定义逐序列指标均值的描述性 bootstrap 区间；它不是 pooled-token AUROC/AUPRC 的置信区间。缺少正负两类的单条序列 AUROC/AP 为 NA，仍纳入 pooled 评价和覆盖率统计。

## 文件与复现

- PROTOCOL.md：推理前固定的规则。
- selected_sequences.csv / selection_manifest.json：800 条最终序列及冻结校验和。
- frozen_readouts.json / *_group_definitions.json / frozen_train_scales.csv：原始固定读出定义。
- download_candidates.py → run_filter.sh → run_inference.py → write_report.py：下载、去重、推理、评价与报告。
- external_validation.ipynb：完整评价代码、方法与本次结果。
- module_metrics.csv / module_per_sequence.csv：全部模块、成员与逐序列结果。
- ploop_metrics.csv / ploop_per_sequence.csv / ploop_per_core.csv：位置特征逐层结果。
- inference_checks_*.json / label_rule_checks.json / evaluation_checks.json / final_verification.json：模型、标签、样本复用与文件完整性检查。

最终 cohort SHA256：`{selection['cohort_sha256']}`。原实验数据和主图未被覆盖。
'''
    (OUT/'REPORT.md').write_text(report)
    # Notebook preserves complete analysis code and displays the already computed result tables.
    def markdown(source):
        return {'cell_type':'markdown','metadata':{},'source':source}
    def code_cell(source):
        return {'cell_type':'code','metadata':{},'source':source,'execution_count':None,'outputs':[]}
    notebook={'nbformat':4,'nbformat_minor':4,'metadata':{},'cells':[]}
    cells=[markdown(report),markdown('## 完整评价代码\n下面单元会使用本目录冻结的 800 条序列和已缓存激活重新计算全部指标，不重新下载或训练。')]
    code=f"__file__ = {str(OUT/'evaluate_external.py')!r}\n"+(OUT/'evaluate_external.py').read_text()
    cells.append(code_cell(code))
    cells.append(code_cell("from IPython.display import display\ndisplay(pd.read_csv(OUT/'fig5b_external_metrics.csv'))\ndisplay(pd.read_csv(OUT/'ploop_summary.csv'))"))
    notebook['cells']=cells
    notebook['metadata']={'kernelspec':{'display_name':'Python 3 (hfds)','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}}
    (OUT/'external_validation.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    manifests={f:json.loads((OUT/f'inference_manifest_{f}.json').read_text()) for f in frame.family.unique()}
    assert all(m['cohort_sha256']==selection['cohort_sha256'] and m['n_sequences']==200 for m in manifests.values())
    assert len({m['sae_sha256'] for m in manifests.values()})==1 and len({m['model_weights_sha256'] for m in manifests.values()})==1
    final=dict(checks,activation_files=len(list((OUT/'activations').glob('*.npz'))),n_tokens=int(frame.length.sum()),n_taxa=frame.groupby('family').taxid.nunique().to_dict(),all_family_inference_manifests_match=True,cohort_checks=json.loads((OUT/'cohort_checks.json').read_text()),original_label_checks=json.loads((OUT/'label_rule_checks.json').read_text()),notebook_code_sha256=hashlib.sha256(code.encode()).hexdigest())
    assert final['activation_files']==800
    (OUT/'final_verification.json').write_text(json.dumps(final,indent=2))
    print(summary.to_string(index=False))
    print('Report and notebook written:',OUT)

if __name__=='__main__':
    main()
