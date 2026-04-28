"""
Objects required to load LSTM embedding pose-correction .keras files from
`LSTM_embedding_pose_correction.ipynb` (Lambda layers with named callables).
"""

from __future__ import annotations

import tensorflow as tf

__all__ = ["timestep_indices", "LSTM_CUSTOM_OBJECTS"]


def timestep_indices(pose):
    """Batch of integer row indices 0..T-1 for Embedding(time). Matches notebook."""
    t = tf.shape(pose)[1]
    b = tf.shape(pose)[0]
    idx = tf.range(t, dtype=tf.int32)
    return tf.tile(idx[None, :], [b, 1])


LSTM_CUSTOM_OBJECTS: dict[str, object] = {
    "timestep_indices": timestep_indices,
}
