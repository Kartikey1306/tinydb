"""
Contains the :class:`base class <tinydb.storages.Storage>` for storages and
implementations.
"""

import contextlib
import errno
import json
import os
import stat
import tempfile
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

__all__ = ('Storage', 'JSONStorage', 'MemoryStorage')


def _fsync_directory(path: str) -> None:
    """
    Flush changes to a directory's entries (like a rename) to disk.

    Best effort: Windows can't open directories, and some filesystems don't
    support fsync on them.
    """
    if os.name == 'nt':
        return

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


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

    Writes are atomic: the new state is written to a temporary file next to
    the database, which then replaces the database file. A crash or a full
    disk during a write leaves the previous state intact instead of a
    half-written file.
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

        # Writes never go through our file handle, so after the first write
        # it is reopened read-only. Reopening in a 'w' mode would truncate
        # the file we just wrote.
        self._read_mode = 'rb' if 'b' in access_mode else 'r'
        self._warned_in_place = False

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

        # Writes replace the file the path points to. Resolve symlinks once,
        # so we replace the real file and keep the link intact.
        self._target = os.path.realpath(self._path)

        # Open the file for reading/writing
        self._handle = open(self._path, mode=self._mode, encoding=encoding)

    def _reopen(self) -> None:
        self._handle.close()
        self._handle = open(self._target, mode=self._read_mode, encoding=self._encoding)

    def _reopen_if_replaced(self) -> None:
        # Another JSONStorage (in this or another process) may have swapped
        # in a new file since ours was opened. Our handle still points at
        # the old, now unlinked file, so follow the path to the current one.
        try:
            current = os.stat(self._target)
        except FileNotFoundError:
            return

        opened = os.fstat(self._handle.fileno())
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            self._reopen()

    def close(self) -> None:
        self._handle.close()

    def read(self) -> Optional[dict[str, dict[str, Any]]]:
        self._reopen_if_replaced()

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
        if not self._write_atomically(serialized):
            self._write_in_place(serialized)

    def _write_atomically(self, serialized: str) -> bool:
        """
        Swap in a new file holding ``serialized``.

        Returns ``False`` if the database file can't be replaced, in which
        case nothing has been changed.
        """
        # The temporary file has to be in the same directory: os.replace()
        # can't move files between filesystems. mkstemp() gives it a unique,
        # unpredictable name and creates it exclusively, so two databases
        # never share a temporary file and nothing can be planted in its
        # place.
        directory = os.path.dirname(self._target)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=directory,
                prefix='.{}.'.format(os.path.basename(self._target)),
                suffix='.tmp',
            )
        except PermissionError:
            # The database file is writable, but its directory isn't
            return False

        try:
            with open(fd, 'w', encoding=self._encoding) as tmp:
                tmp.write(serialized)
                tmp.flush()
                os.fsync(tmp.fileno())

            self._copy_metadata(tmp_path)

            # Windows refuses to replace a file that is still open, so release
            # our handle for the swap. Reopen it whether or not the swap
            # worked, so a failed write doesn't leave the storage unusable.
            self._handle.close()
            try:
                os.replace(tmp_path, self._target)
            finally:
                self._handle = open(self._target, mode=self._read_mode,
                                    encoding=self._encoding)
        except BaseException as e:
            # Don't leave a stray temporary file behind
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)

            # A database file that is a mount point (like a single-file Docker
            # bind mount) can be written to, but not replaced
            if isinstance(e, OSError) and e.errno in (errno.EBUSY, errno.EXDEV):
                return False

            raise

        # Make the rename itself durable, not just the file contents
        _fsync_directory(directory)

        return True

    def _copy_metadata(self, tmp_path: str) -> None:
        """
        Give the temporary file the database file's owner, group and mode.

        mkstemp() creates it as 0600 and owned by us. Changing the owner
        needs root and changing the group needs membership in it, so both are
        best effort.
        """
        try:
            st = os.stat(self._target)
        except FileNotFoundError:
            return

        # chown() may clear setuid/setgid bits, so it goes first
        if hasattr(os, 'chown'):
            for uid, gid in ((st.st_uid, st.st_gid), (-1, st.st_gid)):
                try:
                    os.chown(tmp_path, uid, gid)
                    break
                except OSError:
                    continue

        os.chmod(tmp_path, stat.S_IMODE(st.st_mode))

    def _write_in_place(self, serialized: str) -> None:
        """
        Overwrite the database file directly, for when it can't be replaced.

        This is how all writes used to work. It isn't crash safe, so warn
        about it (once per storage).
        """
        if not self._warned_in_place:
            warnings.warn(
                'Cannot replace {!r} atomically (its directory is not '
                'writable, or it is a mount point). Writing in place instead, '
                'so a crash during a write can corrupt it.'.format(self._path),
                RuntimeWarning,
            )
            self._warned_in_place = True

        with open(self._target, 'r+', encoding=self._encoding) as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())

            # Remove data that is behind the new cursor in case the file has
            # gotten shorter
            f.truncate()


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
