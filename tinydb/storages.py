"""
Contains the :class:`base class <tinydb.storages.Storage>` for storages and
implementations.
"""

import json
import os
import tempfile
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

__all__ = ('Storage', 'JSONStorage', 'MemoryStorage')


def touch(path: str, create_dirs: bool):
    """
    Create a file if it doesn't exist yet.

    :param path: The file to create.
    :param create_dirs: Whether to create all missing parent directories.
    """
    if create_dirs:
        base_dir = os.path.dirname(path)

        # Check if we need to create missing parent directories
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir)

    # Create the file by opening it in 'a' mode which creates the file if it
    # does not exist yet but does not modify its contents
    with open(path, 'a'):
        pass


class Storage(ABC):
    """
    The abstract base class for all Storages.

    A Storage (de)serializes the current state of the database and stores it in
    some place (memory, file on disk, ...).
    """

    # Using ABCMeta as metaclass allows instantiating only storages that have
    # implemented read and write

    @abstractmethod
    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        """
        Read the current state.

        Any kind of deserialization should go here.

        Return ``None`` here to indicate that the storage is empty.
        """

        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def write(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Write the current state of the database to the storage.

        Any kind of serialization should go here.

        :param data: The current state of the database.
        """

        raise NotImplementedError('To be overridden!')

    def close(self) -> None:
        """
        Optional: Close open file handles, etc.
        """

        pass


class JSONStorage(Storage):
    """
    Store the data in a JSON file.

    Writes are atomic: the new state is written to a temporary file which
    then replaces the database file, so a crash or a full disk during a
    write leaves the previous state intact instead of a half-written file.
    """

    def __init__(self, path: str, create_dirs=False, encoding=None, access_mode='r+', **kwargs):
        """
        Create a new instance.

        Also creates the storage file, if it doesn't exist and the access mode
        is appropriate for writing.

        **Note:** Using an access mode other than `r` or `r+` will probably
        lead to data loss or data corruption!

        **Note:** **Never** pass untrusted or user-controlled code as ``kwargs``
        members like ``cls`` or ``default`` will be called on every write
        operation.

        :param path: Where to store the JSON data.
        :param access_mode: mode in which the file is opened (r, r+)
        :type access_mode: str
        """

        super().__init__()

        self._path = os.fspath(path)
        self._mode = access_mode
        self._encoding = encoding
        self.kwargs = kwargs

        if access_mode not in ('r', 'rb', 'r+', 'rb+'):
            warnings.warn(
                'Using an `access_mode` other than \'r\', \'rb\', \'r+\' '
                'or \'rb+\' can cause data loss or corruption'
            )

        # Any of the writing modes
        self._writable = any(character in self._mode for character in ('+', 'w', 'a'))

        # Create the file if it doesn't exist and creating is allowed by the
        # access mode
        if self._writable:
            touch(self._path, create_dirs=create_dirs)

        # Open the file for reading/writing
        self._handle = open(self._path, mode=self._mode, encoding=encoding)

    def close(self) -> None:
        self._handle.close()

    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        # Get the file size by moving the cursor to the file end and reading
        # its location
        self._handle.seek(0, os.SEEK_END)
        size = self._handle.tell()

        if not size:
            # File is empty, so we return ``None`` so TinyDB can properly
            # initialize the database
            return None
        else:
            # Return the cursor to the beginning of the file
            self._handle.seek(0)

            # Load the JSON contents of the file
            return json.load(self._handle)

    def write(self, data: dict[str, dict[str, Any]]):
        # Writes no longer go through our file handle, so its access mode
        # doesn't stop them for us
        if not self._writable:
            raise IOError('Cannot write to the database. Access mode is "{0}"'.format(self._mode))

        # Serialize the database state using the user-provided arguments
        serialized = json.dumps(data, **self.kwargs)

        # Never modify the database file in place: if the process dies or the
        # disk fills up halfway through, the file is left truncated or with
        # stale bytes at the end and can't be parsed anymore. Write the new
        # state to a temporary file first, make sure it has reached the disk
        # and then swap it in with os.replace(), which is atomic. The file
        # always holds either the complete old or the complete new state.
        tmp_path = os.path.join(tempfile.gettempdir(),
                                os.path.basename(self._path) + '.tmp')

        with open(tmp_path, 'w', encoding=self._encoding) as tmp:
            tmp.write(serialized)
            tmp.flush()
            os.fsync(tmp.fileno())

        # Windows refuses to replace a file that is still open, so release
        # our handle first and reopen it on the new file afterwards
        self._handle.close()
        os.replace(tmp_path, self._path)
        self._handle = open(self._path, mode=self._mode, encoding=self._encoding)


class MemoryStorage(Storage):
    """
    Store the data as JSON in memory.
    """

    def __init__(self):
        """
        Create a new instance.
        """

        super().__init__()
        self.memory = None

    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        return self.memory

    def write(self, data: dict[str, dict[str, Any]]):
        self.memory = data
