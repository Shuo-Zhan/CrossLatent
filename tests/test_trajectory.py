"""Scientific invariants of frozen-coefficient prediction trajectories."""
import csv
import json
import numpy as np
import pytest
from crosslatent import LinearReadout, predict_steps, export_steps
from crosslatent.cli import main


def sample():
    model = LinearReadout([2,0], [1.,-1.], 0.,3,'fixed-dictionary',threshold=.5)
    x = np.array([[0,99,1],[1,99,1],[1,99,0]],dtype=float)
    return model,x


def test_cumulative_scores_preserve_frozen_weights_and_final_prediction():
    model,x=sample()
    saved=model.to_dict()
    out=predict_steps(model,x,dictionary_id=model.dictionary_id)
    np.testing.assert_array_equal(out['scores'],[[0,1,1],[0,1,0],[0,0,-1]])
    np.testing.assert_array_equal(out['scores'][:,-1],model.predict(x))
    np.testing.assert_array_equal(out['contributions'],[[1,0],[1,-1],[0,-1]])
    assert model.to_dict()==saved
    assert out['latent_ids'].tolist()==[2,0]


def test_transition_flags_include_suppression_and_actual_predecessor():
    model,x=sample()
    out=predict_steps(model,x,dictionary_id=model.dictionary_id)
    assert out['added'][:,0].tolist()==[True,True,False]
    assert out['retained'][:,1].tolist()==[True,False,False]
    assert out['removed'][:,1].tolist()==[False,True,False]
    baseline_positive=LinearReadout([0],[-1.],1.,1,'d',threshold=.5)
    out=predict_steps(baseline_positive,np.array([[2.]]),dictionary_id='d')
    assert out['labels'].tolist()==[[True,False]]
    assert out['removed'].tolist()==[[True]]


def test_continuous_scores_have_no_threshold_or_binary_labels():
    model=LinearReadout([0],[2.],-3.,1,'d',task='continuous',threshold=None)
    out=predict_steps(model,np.array([[1.],[-2.]]),dictionary_id='d')
    np.testing.assert_array_equal(out['scores'],[[-3,-1],[-3,-7]])
    assert not {'labels','threshold','added','retained','removed'} & out.keys()


@pytest.mark.parametrize('settings,match',[
    ({'dictionary_id':'wrong'},'dictionary_id'),
    ({'lengths':[1,1]},'lengths'),
    ({'lengths':[1.,2.]},'lengths'),
    ({'lengths':[1,0,2]},'lengths'),
    ({'lengths':[1,2],'sequence_ids':['same','same']},'sequence_ids'),
])
def test_invalid_alignment_rejected(settings,match):
    model,x=sample()
    with pytest.raises(ValueError,match=match):
        predict_steps(model,x,**{'dictionary_id':model.dictionary_id,**settings})


def test_cli_export_relocated_inputs_and_sequence_coordinates(tmp_path):
    model,x=sample();model.save(tmp_path/'model.json');np.save(tmp_path/'x.npy',x)
    np.save(tmp_path/'lengths.npy',np.array([2,1]));(tmp_path/'ids.json').write_text('["a", "b"]')
    main(['predict-steps','--model',str(tmp_path/'model.json'),'--activations',str(tmp_path/'x.npy'),
          '--dictionary-id',model.dictionary_id,'--lengths',str(tmp_path/'lengths.npy'),
          '--sequence-ids',str(tmp_path/'ids.json'),'--output',str(tmp_path/'steps')])
    with np.load(tmp_path/'steps/trajectory.npz',allow_pickle=False) as out:
        np.testing.assert_array_equal(out['scores'][:,-1],model.predict(x))
        assert out['offsets'].tolist()==[0,2,3]
    with (tmp_path/'steps/positions.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==9
    assert rows[-1]['sequence_id']=='b' and rows[-1]['position']=='1'
    assert rows[-1]['latent_id']=='0' and rows[-1]['score']=='-1.0'
    assert rows[5]['change']=='removed'
    with pytest.raises(FileExistsError):
        export_steps(model,x,tmp_path/'steps',dictionary_id=model.dictionary_id)


def test_empty_and_nonfinite_selected_responses_rejected():
    model,x=sample()
    with pytest.raises(ValueError,match='at least one'):
        predict_steps(model,x[:0],dictionary_id=model.dictionary_id)
    x[0,2]=np.nan
    with pytest.raises(ValueError,match='finite'):
        predict_steps(model,x,dictionary_id=model.dictionary_id)


def test_final_threshold_labels_match_existing_predict_at_rounding_boundary():
    rng=np.random.default_rng(73)
    x=rng.normal(size=(30,10))
    model=LinearReadout(list(range(10)),rng.normal(size=10).tolist(),.137,10,'d')
    final=model.predict(x)
    model.threshold=float(final[0])
    out=predict_steps(model,x,dictionary_id='d')
    np.testing.assert_array_equal(out['scores'][:,-1],final)
    np.testing.assert_array_equal(out['labels'][:,-1],final>=model.threshold)
