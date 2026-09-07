from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_segpose.evaluation.visualizer import (
    _greedy_mask_matches,
    generate_comparison_board,
)


def test_greedy_mask_matches_is_one_to_one():
    mask = np.zeros((40, 40), np.uint8)
    mask[5:25, 5:25] = 255
    prediction = type("Prediction", (), {"_mask": mask})()
    assert _greedy_mask_matches([mask, mask], [prediction], 0.30) == [(1, 0)]


def test_comparison_board_validates_missing_dataset(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        generate_comparison_board(tmp_path / "missing", tmp_path / "board.png")


def test_comparison_board_validates_algorithms(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="Unsupported algorithms"):
        generate_comparison_board(tmp_path, tmp_path / "board.png", algorithms=["unknown"])
