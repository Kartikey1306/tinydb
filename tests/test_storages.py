import errno
import json
import os
import random
import stat
import tempfile
import warnings

import pytest

from tinydb import TinyDB, where
from tinydb.storages import JSONStorage, MemoryStorage, Storage, touch
from tinydb.table import Document

random.seed()

doc = {'none': [None, None], 'int': 42, 'float': 3.1415899999999999,
       'list': ['LITE', 'RES_ACID', 'SUS_DEXT'],
       'dict': {'hp': 13, 'sp': 5},
       'bool': [True, False, True, False]}


def test_json(tmpdir):
    # Write contents
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    storage.write(doc)

    # Verify contents
    assert doc == storage.read()
    storage.close()


def test_json_kwargs(tmpdir):
    db_file = tmpdir.join('test.db')
    db = TinyDB(str(db_file), sort_keys=True, indent=4, separators=(',', ': '))

    # Write contents
    db.insert({'b': 1})
    db.insert({'a': 1})

    assert db_file.read() == '''{
    "_default": {
        "1": {
            "b": 1
        },
        "2": {
            "a": 1
        }
    }
}'''
    db.close()


def test_json_readwrite(tmpdir):
    """
    Regression test for issue #1
    """
    path = str(tmpdir.join('test.db'))

    # Create TinyDB instance
    db = TinyDB(path, storage=JSONStorage)

    item = {'name': 'A very long entry'}
    item2 = {'name': 'A short one'}

    def get(s):
        return db.get(where('name') == s)

    db.insert(item)
    assert get('A very long entry') == item

    db.remove(where('name') == 'A very long entry')
    assert get('A very long entry') is None

    db.insert(item2)
    assert get('A short one') == item2

    db.remove(where('name') == 'A short one')
    assert get('A short one') is None

    db.close()


def test_json_read(tmpdir):
    r"""Open a database only for reading"""
    path = str(tmpdir.join('test.db'))
    with pytest.raises(FileNotFoundError):
        db = TinyDB(path, storage=JSONStorage, access_mode='r')
    # Create small database
    db = TinyDB(path, storage=JSONStorage)
    db.insert({'b': 1})
    db.insert({'a': 1})
    db.close()
    # Access in read mode
    db = TinyDB(path, storage=JSONStorage, access_mode='r')
    assert db.get(where('a') == 1) == {'a': 1}  # reading is fine
    with pytest.raises(IOError):
        db.insert({'c': 1})  # writing is not
    db.close()


def test_json_write_failure_keeps_previous_state(tmpdir, monkeypatch):
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    storage.write(doc)

    # Fail after the new (shorter) state has been written out but before it
    # is known to be on disk, like a dying disk or a crash would
    def failing_fsync(fd):
        raise OSError(errno.EIO, 'simulated I/O error')

    monkeypatch.setattr(os, 'fsync', failing_fsync)
    with pytest.raises(OSError):
        storage.write({'_default': {}})
    monkeypatch.undo()

    # The database file must still hold the complete previous state
    with open(path) as f:
        assert json.load(f) == doc

    storage.close()


def test_json_storage_usable_across_writes(tmpdir):
    # Every write swaps in a new file, so the storage must keep reading the
    # current one rather than the file it opened initially
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)

    for i in range(5):
        data = {'_default': {str(n): {'n': n} for n in range(5 - i)}}
        storage.write(data)
        assert storage.read() == data

    storage.close()

    with open(path) as f:
        assert json.load(f) == {'_default': {'0': {'n': 0}}}


def test_json_failed_write_cleans_up(tmpdir, monkeypatch):
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    storage.write(doc)

    def failing_fsync(fd):
        raise OSError(errno.ENOSPC, 'simulated full disk')

    monkeypatch.setattr(os, 'fsync', failing_fsync)
    with pytest.raises(OSError):
        storage.write({'_default': {}})
    monkeypatch.undo()

    # No temporary file is left behind, and the storage is still usable
    assert os.listdir(str(tmpdir)) == ['test.db']
    assert storage.read() == doc
    storage.close()


def test_json_failed_replace_keeps_storage_usable(tmpdir, monkeypatch):
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    storage.write(doc)

    def failing_replace(src, dst):
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(os, 'replace', failing_replace)
    with pytest.raises(OSError):
        storage.write({'_default': {}})
    monkeypatch.undo()

    assert storage.read() == doc
    assert os.listdir(str(tmpdir)) == ['test.db']

    storage.write({'_default': {}})
    assert storage.read() == {'_default': {}}
    storage.close()


def test_json_temp_file_next_to_database(tmpdir, monkeypatch):
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    sources = []

    real_replace = os.replace

    def recording_replace(src, dst):
        sources.append(src)
        real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', recording_replace)
    storage.write(doc)
    storage.write(doc)
    storage.close()

    # Same directory (and so the same filesystem) as the database, with a
    # fresh name for every write
    assert [os.path.dirname(src) for src in sources] == [str(tmpdir)] * 2
    assert sources[0] != sources[1]


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permissions')
def test_json_write_keeps_file_mode(tmpdir):
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    storage.write(doc)

    for mode in (0o600, 0o640):
        os.chmod(path, mode)
        storage.write(doc)
        assert stat.S_IMODE(os.stat(path).st_mode) == mode

    storage.close()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX groups')
def test_json_write_keeps_group(tmpdir):
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    storage.write(doc)

    current = os.stat(path).st_gid
    others = [gid for gid in os.getgroups() if gid != current]
    if not others:
        pytest.skip('needs membership in a second group')

    os.chown(path, -1, others[0])
    storage.write(doc)
    storage.close()

    assert os.stat(path).st_gid == others[0]


@pytest.mark.skipif(os.name == 'nt' or os.geteuid() == 0,
                    reason='POSIX permissions, not as root')
def test_json_unwritable_directory_falls_back_to_in_place(tmpdir, monkeypatch):
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    storage.write(doc)

    os.chmod(str(tmpdir), 0o500)
    try:
        with pytest.warns(RuntimeWarning, match='not writable'):
            storage.write({'_default': {}})
    finally:
        os.chmod(str(tmpdir), 0o700)

    assert storage.read() == {'_default': {}}
    assert os.listdir(str(tmpdir)) == ['test.db']

    # Once the directory is writable again, writes are atomic again
    replaced = []
    real_replace = os.replace

    def recording_replace(src, dst):
        replaced.append(src)
        real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', recording_replace)
    storage.write(doc)
    monkeypatch.undo()

    assert len(replaced) == 1
    assert storage.read() == doc
    storage.close()


def test_json_mount_point_falls_back_to_in_place(tmpdir, monkeypatch):
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    storage.write(doc)

    # What os.replace() raises when the target is a bind-mounted file
    attempts = []

    def busy(src, dst):
        attempts.append(src)
        raise OSError(errno.EBUSY, os.strerror(errno.EBUSY))

    monkeypatch.setattr(os, 'replace', busy)
    with pytest.warns(RuntimeWarning, match='mount point'):
        storage.write({'_default': {}})

    # Later writes go straight to the fallback: no new temp file, no
    # repeated swap attempt and no repeated warning
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        storage.write({'_default': {'1': {'a': 1}}})
    monkeypatch.undo()

    assert len(attempts) == 1

    assert storage.read() == {'_default': {'1': {'a': 1}}}
    assert os.listdir(str(tmpdir)) == ['test.db']
    storage.close()


def test_json_fallback_refused_when_warnings_are_errors(tmpdir, monkeypatch):
    path = str(tmpdir.join('test.db'))
    storage = JSONStorage(path)
    storage.write(doc)

    def busy(src, dst):
        raise OSError(errno.EBUSY, os.strerror(errno.EBUSY))

    monkeypatch.setattr(os, 'replace', busy)

    # With warnings as errors, falling back to a non-atomic write is refused,
    # and keeps being refused rather than quietly going ahead next time
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for _ in range(2):
            with pytest.raises(RuntimeWarning):
                storage.write({'_default': {}})
    monkeypatch.undo()

    assert storage.read() == doc
    storage.close()


@pytest.mark.skipif(not hasattr(os, 'symlink') or os.name == 'nt',
                    reason='needs symlinks')
def test_json_write_through_symlink(tmpdir):
    real = str(tmpdir.join('real.db'))
    link = str(tmpdir.join('link.db'))
    with open(real, 'w'):
        pass
    os.symlink(real, link)

    storage = JSONStorage(link)
    storage.write(doc)
    storage.close()

    # The link is kept and the data ends up in the file it points to
    assert os.path.islink(link)
    with open(real) as f:
        assert json.load(f) == doc


def test_json_write_mode_w_keeps_data(tmpdir):
    path = str(tmpdir.join('test.db'))
    with pytest.warns(UserWarning):
        storage = JSONStorage(path, access_mode='w')
    storage.write(doc)
    storage.close()

    with open(path) as f:
        assert json.load(f) == doc


def test_json_sees_writes_from_other_instance(tmpdir):
    path = str(tmpdir.join('test.db'))
    writer = JSONStorage(path)
    reader = JSONStorage(path)

    writer.write(doc)
    assert reader.read() == doc

    writer.write({'_default': {}})
    assert reader.read() == {'_default': {}}

    writer.close()
    reader.close()


def test_create_dirs():
    temp_dir = tempfile.gettempdir()

    while True:
        dname = os.path.join(temp_dir, str(random.getrandbits(20)))
        if not os.path.exists(dname):
            db_dir = dname
            db_file = os.path.join(db_dir, 'db.json')
            break

    with pytest.raises(IOError):
        JSONStorage(db_file)

    JSONStorage(db_file, create_dirs=True).close()
    assert os.path.exists(db_file)

    # Use create_dirs with already existing directory
    JSONStorage(db_file, create_dirs=True).close()
    assert os.path.exists(db_file)

    os.remove(db_file)
    os.rmdir(db_dir)


def test_json_invalid_directory():
    with pytest.raises(IOError):
        with TinyDB('/this/is/an/invalid/path/db.json', storage=JSONStorage):
            pass


def test_in_memory():
    # Write contents
    storage = MemoryStorage()
    storage.write(doc)

    # Verify contents
    assert doc == storage.read()

    # Test case for #21
    other = MemoryStorage()
    other.write({})
    assert other.read() != storage.read()


def test_in_memory_close():
    with TinyDB(storage=MemoryStorage) as db:
        db.insert({})


def test_custom():
    # noinspection PyAbstractClass
    class MyStorage(Storage):
        pass

    with pytest.raises(TypeError):
        MyStorage()


def test_read_once():
    count = 0

    # noinspection PyAbstractClass
    class MyStorage(Storage):
        def __init__(self):
            self.memory = None

        def read(self):
            nonlocal count
            count += 1

            return self.memory

        def write(self, data):
            self.memory = data

    with TinyDB(storage=MyStorage) as db:
        assert count == 0

        db.table(db.default_table_name)

        assert count == 0

        db.all()

        assert count == 1

        db.insert({'foo': 'bar'})

        assert count == 3  # One for getting the next ID, one for the insert

        db.all()

        assert count == 4


def test_custom_with_exception():
    class MyStorage(Storage):
        def read(self):
            pass

        def write(self, data):
            pass

        def __init__(self):
            raise ValueError()

        def close(self):
            raise RuntimeError()

    with pytest.raises(ValueError):
        with TinyDB(storage=MyStorage) as db:
            pass


def test_yaml(tmpdir):
    """
    :type tmpdir: py._path.local.LocalPath
    """

    try:
        import yaml
    except ImportError:
        return pytest.skip('PyYAML not installed')

    def represent_doc(dumper, data):
        # Represent `Document` objects as their dict's string representation
        # which PyYAML understands
        return dumper.represent_data(dict(data))

    yaml.add_representer(Document, represent_doc)

    class YAMLStorage(Storage):
        def __init__(self, filename):
            self.filename = filename
            touch(filename, False)

        def read(self):
            with open(self.filename) as handle:
                data = yaml.safe_load(handle.read())
                return data

        def write(self, data):
            with open(self.filename, 'w') as handle:
                yaml.dump(data, handle)

        def close(self):
            pass

    # Write contents
    path = str(tmpdir.join('test.db'))
    db = TinyDB(path, storage=YAMLStorage)
    db.insert(doc)
    assert db.all() == [doc]

    db.update({'name': 'foo'})

    assert '!' not in tmpdir.join('test.db').read()

    assert db.contains(where('name') == 'foo')
    assert len(db) == 1


def test_encoding(tmpdir):
    japanese_doc = {"Test": u"こんにちは世界"}

    path = str(tmpdir.join('test.db'))
    # cp936 is used for japanese encodings
    jap_storage = JSONStorage(path, encoding="cp936")
    jap_storage.write(japanese_doc)

    try:
        exception = json.decoder.JSONDecodeError
    except AttributeError:
        exception = ValueError

    with pytest.raises(exception):
        # cp037 is used for english encodings
        eng_storage = JSONStorage(path, encoding="cp037")
        eng_storage.read()

    jap_storage = JSONStorage(path, encoding="cp936")
    assert japanese_doc == jap_storage.read()


def test_json_invalid_mode_warning(tmpdir):
    path = str(tmpdir.join('test.db'))
    with pytest.warns(UserWarning, match='Using an `access_mode` other than'):
        JSONStorage(path, access_mode='w')
