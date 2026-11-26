"""
Behavioural tests for the 48px OCR beam search early stop
(infer_beam_batch_tensor, the path the 48px OCR uses at inference).

The model is built with random weights (no checkpoint download). The output
bias is rigged so the end token always wins, which means every hypothesis is
finished after its first real token and the search should stop right away.
"""
import pytest
import torch

from manga_translator.config import OcrConfig
from manga_translator.ocr.model_48px import OCR

START_TOK, END_TOK = 1, 2
MAX_SEQ_LENGTH = 64
DICTIONARY = ['<PAD>', '<S>', '</S>', '<SP>', 'a', 'b', 'c', 'd']


def build_model():
    torch.manual_seed(0)
    model = OCR(DICTIONARY, 255)
    model.eval()
    with torch.no_grad():
        model.pred.bias.zero_()
        model.pred.bias[END_TOK] = 1e4
    return model


def count_decoder_steps(model):
    calls = {'n': 0}
    original = model.decoders.forward

    def counting_forward(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)

    model.decoders.forward = counting_forward
    return calls


def make_batch(n=2, width=64):
    img = torch.rand(n, 3, 48, width) * 2 - 1
    return img, [width] * n


@pytest.mark.parametrize('beams_k', [1, 2, 5])
def test_infer_beam_batch_tensor_stops_early(beams_k):
    model = build_model()
    steps = count_decoder_steps(model)
    img, widths = make_batch()
    with torch.no_grad():
        result = model.infer_beam_batch_tensor(img, widths, beams_k=beams_k, max_seq_length=MAX_SEQ_LENGTH)

    assert len(result) == 2
    for pred_chars_index, prob, *_ in result:
        # Start token is stripped, so the sequence is just the end token(s).
        assert pred_chars_index[0].item() == END_TOK
        assert len(pred_chars_index) <= 2
        assert prob > 0.5
    # One decoder call for the first token, one for the step that finishes.
    # Before the fix, a beam of 1 ran to MAX_SEQ_LENGTH decoder steps.
    assert steps['n'] <= 2


def test_ocr_config_beams_k():
    assert OcrConfig().beams_k == 5
    assert OcrConfig(beams_k=1).beams_k == 1
    with pytest.raises(ValueError):
        OcrConfig(beams_k=0)
