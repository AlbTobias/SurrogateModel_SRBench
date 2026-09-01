import numpy as np
import pytest

from experiment.evaluate_surrogate import normalize_prediction_shape


@pytest.mark.parametrize("prediction", [2.5, np.array(2.5), np.array([2.5])])
def test_scalar_prediction_is_broadcast(prediction):
    values, broadcast = normalize_prediction_shape(prediction, 4)

    np.testing.assert_array_equal(values, np.array([2.5, 2.5, 2.5, 2.5]))
    assert broadcast is True


def test_complete_prediction_vector_is_unchanged():
    prediction = np.array([[1.0], [2.0], [3.0]])

    values, broadcast = normalize_prediction_shape(prediction, 3)

    np.testing.assert_array_equal(values, np.array([1.0, 2.0, 3.0]))
    assert broadcast is False


def test_other_prediction_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="expected 4, received 2"):
        normalize_prediction_shape(np.array([1.0, 2.0]), 4)
