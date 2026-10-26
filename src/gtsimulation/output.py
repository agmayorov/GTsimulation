from enum import Enum

import h5py
import numpy as np


def _write_value(parent, name, value, compression):
    """Write a value from the simulation result to an HDF5 group."""
    if isinstance(value, dict):
        group = parent.create_group(name)
        for key, item in value.items():
            _write_value(group, str(key), item, compression)
        return

    if isinstance(value, (list, tuple)):
        group = parent.create_group(name)
        group.attrs["python_kind"] = "tuple" if isinstance(value, tuple) else "list"
        for index, item in enumerate(value):
            _write_value(group, str(index), item, compression)
        return

    if value is None:
        parent.create_group(name).attrs["python_kind"] = "none"
        return

    if isinstance(value, Enum):
        value = value.name
    if isinstance(value, str):
        parent.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
        return

    array = np.asarray(value)
    options = {}
    if compression is not None and array.ndim > 0 and array.size > 0:
        options["compression"] = compression
    parent.create_dataset(name, data=array, **options)


class _HDF5EventWriter:
    """Collect and write one event without keeping its complete track in RAM."""

    def __init__(self, group, save_codes, chunk_size, compression):
        self.group = group
        self.track_group = group.create_group("Track")
        self.save_codes = {
            key: code for key, code in save_codes.items() if code is not None
        }
        self.chunk_size = chunk_size
        self.compression = compression
        self.buffer = []

    def append(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_size:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return

        rows = np.asarray(self.buffer)
        for key, code in self.save_codes.items():
            values = rows[:, code]
            if key not in self.track_group:
                tail_shape = values.shape[1:]
                self.track_group.create_dataset(
                    key,
                    shape=(0, *tail_shape),
                    maxshape=(None, *tail_shape),
                    chunks=(self.chunk_size, *tail_shape),
                    dtype=values.dtype,
                    compression=self.compression,
                )

            dataset = self.track_group[key]
            start = dataset.shape[0]
            dataset.resize(start + len(values), axis=0)
            dataset[start:] = values

        self.buffer.clear()

    def finish(self, result):
        self._flush()

        # Track is empty when it has already been streamed. If calculating
        # additions required the complete track, write it here instead.
        for key, value in result["Track"].items():
            _write_value(self.track_group, key, value, self.compression)
        for key, value in result.items():
            if key != "Track":
                _write_value(self.group, key, value, self.compression)

        self.group.attrs["complete"] = True


class HDF5Writer:
    """Write simulation events to HDF5 as they are calculated."""

    def __init__(self, path, save_codes, chunk_size=4096, compression=None):
        self.file = h5py.File(path, "w")
        self.file.attrs["complete"] = False
        self.events = self.file.create_group("events")
        self.save_codes = save_codes
        self.chunk_size = chunk_size
        self.compression = compression

    def start_event(self, index):
        group = self.events.create_group(str(index))
        group.attrs["complete"] = False
        return _HDF5EventWriter(
            group,
            self.save_codes,
            self.chunk_size,
            self.compression,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.file.attrs["complete"] = True
        self.file.close()
        return False
