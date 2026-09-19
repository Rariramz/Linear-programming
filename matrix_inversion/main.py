from pathlib import Path
import sys

_repository_root = str(Path(__file__).resolve().parents[1])
if _repository_root not in sys.path:
    sys.path.insert(0, _repository_root)

from matrix_inversion.core import (
    input_float_matrix,
    input_float_vector,
    matrix_inversion,
    matrix_multiplication,
    run_example,
)


if __name__ == '__main__':
    run_example()
