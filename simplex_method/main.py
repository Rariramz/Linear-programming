from pathlib import Path
import runpy
import sys


_repository_root = str(Path(__file__).resolve().parents[1])
if _repository_root not in sys.path:
    sys.path.insert(0, _repository_root)

from simplex_method.core import (
    create_matrix_ab,
    create_vector_cb,
    create_vector_theta,
    create_vector_x,
    find_min_delta,
    input_float_matrix,
    input_float_vector,
    input_int_vector,
    main_stage_simplex_method,
)


if __name__ == '__main__':
    runpy.run_module('simplex_method.core', run_name='__main__')
