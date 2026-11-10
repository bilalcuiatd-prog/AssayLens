import pandas as pd
import pytest
from assaylens.ppi_huri import normalize_huri

SEQ_A='ACDEFGHIKLMNPQRSTVWY'*3
SEQ_B='YWVTSRQPNMLKIHGFEDCA'*3


def test_preserves_positive_only_network():
    raw=pd.DataFrame({'Protein1':[SEQ_A,SEQ_A], 'Protein2':[SEQ_B,SEQ_B+'A'], 'Y':[1,1]})
    df,audit=normalize_huri(raw,max_rows=100)
    assert len(df)==2
    assert df.label.eq(1).all()
    assert audit['trainable_binary_labels'] is False
    assert not audit['reference_controls_included']


def test_accepts_explicit_reference_controls_but_flags_them():
    raw=pd.DataFrame({'Protein1':[SEQ_A,SEQ_A], 'Protein2':[SEQ_B,SEQ_B+'A'], 'Y':[1,0]})
    df,audit=normalize_huri(raw,max_rows=100,include_reference_controls=True)
    assert audit['trainable_binary_labels'] is True
    assert set(df.label)=={0,1}
    assert 'not manually confirmed' in ' '.join(audit['limitations'])


def test_excludes_multisequence_and_invalid_labels():
    raw=pd.DataFrame({'Protein1':[SEQ_A, SEQ_A+'*'+SEQ_B,SEQ_A], 'Protein2':[SEQ_B]*3, 'Y':[1,1,99]})
    df,audit=normalize_huri(raw,max_rows=100)
    assert len(df)==1
    assert audit['ambiguous_or_invalid_rows']==2


def test_rejects_empty_or_underspecified_data():
    with pytest.raises(ValueError):
        normalize_huri(pd.DataFrame(),100)
    with pytest.raises(ValueError):
        normalize_huri(pd.DataFrame({'Protein1':['AA'],'Protein2':['AA'],'Y':[1]}),100)
