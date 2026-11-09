"""Read-only exact full-sequence audit for every Fig2 training source and holdout."""
from pathlib import Path
import csv,json,hashlib
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
NORMALIZE=lambda s:''.join(s.split()).upper()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def csvread(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(name,rows):
 if not rows:(OUT/name).write_text('');return
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
manifest=[];data={};idx={};ids={};dataset_summary=[]
def load(name,p):
 raw=csvread(p);records=[];byseq=defaultdict(list);byid=defaultdict(list)
 for rownum,row in enumerate(raw):
  seq=NORMALIZE(row['sequence']);mask=row['mask_protein_domains'];assert seq and len(seq)==len(mask),(name,rownum)
  assert set(mask)<=set('01'),(name,rownum)
  item={'row':rownum,'id':row['uniprot_id'],'sequence':seq,'mask':mask};records.append(item);byseq[seq].append(item);byid[item['id']].append(item)
 data[name]=records;idx[name]=byseq;ids[name]=byid
 manifest.append({'dataset':name,'path':str(p),'sha256':digest(p),'n_rows':len(raw),'unique_sequences':len(byseq)})
 dataset_summary.append({'dataset':name,'n_rows':len(raw),'unique_sequences':len(byseq),'duplicate_rows_within':len(raw)-len(byseq),'unique_ids':len(byid)})
groups=csvread(ROOT/'fig/paper_figures/CrossLatent_Fig2_Group3_clean_2026-10-06/source_data/A_groups.csv');models={r['experiment']:r for r in csvread(ROOT/'fig/paper_figures/fig2/same_clan_evaluation/experiment_manifest.csv')}
for r in groups:
 p=ROOT/r['input_csv'];cfg=json.loads(Path(models[r['source']]['base_json']).read_text())['config'];assert Path(cfg['csv_path']).resolve()==p.resolve(),r['source'];load(r['source'],p)
evalpaths={'original_valid':ROOT/'fig/paper_figures/fig2/data/CL0023_filtered_valid.csv','original_test':ROOT/'fig/paper_figures/fig2/data/CL0023_filtered_test.csv','PF00069':ROOT/'fig/paper_figures/fig2/cross_clan/data/PF00069_filtered.csv','PF04055':ROOT/'fig/paper_figures/fig2/cross_clan/data/PF04055_filtered.csv','group3_clean_valid':ROOT/'fig/experiments/fig2_group3_clean_reselection_20261006/splits/valid.csv','group3_clean_test':ROOT/'fig/experiments/fig2_group3_clean_reselection_20261006/splits/test.csv'}
for name,p in evalpaths.items():load(name,p)
summary=[];matches=[];conflicts=[];id_only=[]
def compare(left,right):
 shared=set(idx[left])&set(idx[right]);sharedids=set(ids[left])&set(ids[right]);confseq=set();confrows=set();pairs=0
 for seq in sorted(shared):
  a=idx[left][seq];b=idx[right][seq];h=hashlib.sha256(seq.encode()).hexdigest();bad=False
  for x in a:
   for y in b:
    pairs+=1
    if x['mask']!=y['mask']:
     bad=True;confseq.add(seq);confrows.add(y['row']);positions=[i for i,(v,w) in enumerate(zip(x['mask'],y['mask'])) if v!=w]
     conflicts.append({'left':left,'right':right,'left_row_zero_based':x['row'],'right_row_zero_based':y['row'],'left_id':x['id'],'right_id':y['id'],'sequence_sha256':h,'length':len(seq),'mask_mismatch_count':len(positions),'positions_zero_based':json.dumps(positions),'left_mask':x['mask'],'right_mask':y['mask']})
  matches.append({'left':left,'right':right,'sequence_sha256':h,'length':len(seq),'left_rows_zero_based':json.dumps([x['row'] for x in a]),'right_rows_zero_based':json.dumps([y['row'] for y in b]),'left_ids':json.dumps([x['id'] for x in a]),'right_ids':json.dumps([y['id'] for y in b]),'any_domain_mask_conflict':bad})
 for ident in sorted(sharedids):
  aset={r['sequence'] for r in ids[left][ident]};bset={r['sequence'] for r in ids[right][ident]}
  if not aset&bset:id_only.append({'left':left,'right':right,'uniprot_id':ident,'left_rows':json.dumps([r['row'] for r in ids[left][ident]]),'right_rows':json.dumps([r['row'] for r in ids[right][ident]])})
 affected_left=sum(len(idx[left][s]) for s in shared);affected_right=sum(len(idx[right][s]) for s in shared)
 result={'left':left,'right':right,'left_rows':len(data[left]),'right_rows':len(data[right]),'shared_unique_sequences':len(shared),'left_affected_rows':affected_left,'right_affected_rows':affected_right,'right_affected_fraction':affected_right/len(data[right]),'matching_row_pairs':pairs,'shared_ids':len(sharedids),'conflicting_unique_sequences':len(confseq),'conflicting_right_rows':len(confrows)};summary.append(result);return result
for g in groups:
 for ev in evalpaths:compare(g['source'],ev)
for a,b in [('original_valid','original_test'),('group3_clean_valid','group3_clean_test'),('original_valid','PF00069'),('original_valid','PF04055'),('original_test','PF00069'),('original_test','PF04055'),('PF00069','PF04055')]:compare(a,b)
# Simulate train > validation > test precedence per source without modifying any input.
clean_counts=[]
for g in groups:
 source=g['source'];training=set(idx[source]);valid=[r for r in data['original_valid'] if r['sequence'] not in training];validseq={r['sequence'] for r in valid};test=[r for r in data['original_test'] if r['sequence'] not in training|validseq]
 clean_counts.append({'source':source,'original_valid':len(data['original_valid']),'valid_removed_train':len(data['original_valid'])-len(valid),'valid_retained':len(valid),'original_test':len(data['original_test']),'test_removed_train':sum(r['sequence'] in training for r in data['original_test']),'test_removed_valid_only':sum(r['sequence'] not in training and r['sequence'] in validseq for r in data['original_test']),'test_retained':len(test)})
write('overlap_summary.csv',summary);write('overlapping_sequences.csv',matches);write('mask_conflicts.csv',conflicts);write('same_id_different_sequences.csv',id_only);write('dataset_summary.csv',dataset_summary);write('per_source_cleaning_simulation.csv',clean_counts)
(OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
assert all(digest(Path(m['path']))==m['sha256'] for m in manifest)
assert next(r for r in summary if r['left']=='group3' and r['right']=='original_valid')['shared_unique_sequences']==26
assert next(r for r in summary if r['left']=='group3' and r['right']=='original_test')['shared_unique_sequences']==18
assert all(next(r for r in summary if r['left']=='group3' and r['right']==ev)['shared_unique_sequences']==0 for ev in ['group3_clean_valid','group3_clean_test'])
report=['# Fig2全部训练来源与留出集的精确序列重叠核查','日期：2026-10-07。只读核查；未删除数据、改标签、重算指标或修改图/稿。','## 规则与输入','规范化完整序列：移除空白、转为大写；不替换残基、不按局部片段匹配。训练文件来自实际冻结模型config并与Fig2A来源表交叉核对。原validation/test为291/290条；Group3当前清理后两集合均262条。行号从0起；唯一序列数与受影响记录数分别汇报。', '## 原validation/test与各训练来源','| 来源 | 训练条数 | valid重叠唯一序列 | valid受影响记录 | test重叠唯一序列 | test受影响记录 |','|---|---:|---:|---:|---:|---:|']
for g in groups:
 a=next(r for r in summary if r['left']==g['source'] and r['right']=='original_valid');b=next(r for r in summary if r['left']==g['source'] and r['right']=='original_test');report.append(f"| {g['label']} | {a['left_rows']} | {a['shared_unique_sequences']} | {a['right_affected_rows']} | {b['shared_unique_sequences']} | {b['right_affected_rows']} |")
report+=['## Group3清理后集合与其他训练来源','| 来源 | clean valid重叠唯一/记录 | clean test重叠唯一/记录 |','|---|---:|---:|']
for g in groups:
 a=next(r for r in summary if r['left']==g['source'] and r['right']=='group3_clean_valid');b=next(r for r in summary if r['left']==g['source'] and r['right']=='group3_clean_test');report.append(f"| {g['label']} | {a['shared_unique_sequences']}/{a['right_affected_rows']} | {b['shared_unique_sequences']}/{b['right_affected_rows']} |")
report+=['## 其他检查']
for a,b in [('original_valid','original_test'),('group3_clean_valid','group3_clean_test')]:
 r=next(x for x in summary if x['left']==a and x['right']==b);report.append(f"- {a} / {b}：{r['shared_unique_sequences']}条唯一重叠序列；左右受影响记录{r['left_affected_rows']}/{r['right_affected_rows']}；掩码不一致的唯一序列{r['conflicting_unique_sequences']}条。")
external=[r for r in summary if r['left'] in {g['source'] for g in groups} and r['right'] in ['PF00069','PF04055']];report.append(f"- 各训练来源与两个外家族：{len(external)}组检查，合计{sum(r['shared_unique_sequences'] for r in external)}条唯一序列交集（跨比较可能重复计数）；逐组见overlap_summary.csv。")
report.append(f'- 掩码冲突逐对记录共{len(conflicts)}条，见mask_conflicts.csv。记录数允许同一序列出现在多个来源比较中。')
report+=['## 后续使用','per_source_cleaning_simulation.csv给出各来源按train > validation > test优先级清理后的预计条数，仅模拟，不写出新的划分或重新选参。若有训练/validation交集，需要清理validation并重新选K/阈值；若仅validation/test重复，也应按上述优先级清理再评价。Group3清理后的集合不能未经核查直接用于其他来源。外家族表现、标签一致性和当前指标是否受影响应分别判断。此检查是精确完整序列交集检查，同源相似性未在本轮计算。','## 交付','overlap_summary.csv：全部73组序列/ID/掩码交集；overlapping_sequences.csv：重叠序列hash、ID、源行；mask_conflicts.csv：冲突标签及位置；same_id_different_sequences.csv：ID重叠但序列不相同；input_manifest.json：输入路径与hash；per_source_cleaning_simulation.csv：逐来源清理规模模拟。']
(OUT/'Fig2_全部来源序列重叠核查_2026-10-07.md').write_text('\n\n'.join(report)+'\n')
(OUT/'verification.json').write_text(json.dumps({'status':'PASS','datasets':len(data),'comparisons':len(summary),'inputs_unchanged':True,'sequence_rule':'full sequence whitespace stripped and uppercased','group3_original_and_clean_counts_reproduced':True},indent=2))
print(json.dumps(clean_counts,indent=2));print('Conflicts:',len(conflicts),'ID-only:',len(id_only))
