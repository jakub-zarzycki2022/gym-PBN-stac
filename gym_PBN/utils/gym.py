import numpy as np

from gym.spaces import Space


class DiscreteRange(Space):
    r"""A discrete space in :math:`\{ start, \\dots, stop \}`.

    Example::

        >>> DiscreteRange(1,16)

    """

    def __init__(self, start, stop) -> None:
        assert stop > start
        self.start = start
        self.stop = stop
        self.n = (stop + 1) - start
        super().__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.start, self.stop + 1)

    def contains(self, x) -> bool:
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)
        else:
            return False
        return as_int >= self.start and as_int <= self.stop

    def __repr__(self):
        return "DiscreteRange(%d,%d)" % self.start, self.stop

    def __eq__(self, other):
        return (
            isinstance(other, DiscreteRange)
            and self.start == other.start
            and self.stop == other.stop
        )
