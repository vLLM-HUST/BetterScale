import pytest
from numerical_trace import decoder_layers

@pytest.mark.parametrize('prefix',['model','language_model.model'])
def test_text_and_multimodal_wrapper_names(prefix):
    layer=object()
    assert decoder_layers([(prefix+'.layers.0',layer),
        (prefix+'.layers.0.mlp',object()),(prefix+'.layers',object())])=={0:layer}

def test_ambiguous_target_rejected():
    with pytest.raises(ValueError,match='Ambiguous'):
        decoder_layers([('model.layers.0',object()),('draft.model.layers.0',object())])
