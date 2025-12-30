import cosmic
from cosmic import Cosmic


def test_imports():
    assert cosmic.__version__
    assert Cosmic is not None
