import os
import threading

import pytest

from tinydb import TinyDB
from tinydb.middlewares import CachingMiddleware, LockingMiddleware
from tinydb.storages import MemoryStorage, JSONStorage

doc = {'none': [None, None], 'int': 42, 'float': 3.1415899999999999,
       'list': ['LITE', 'RES_ACID', 'SUS_DEXT'],
       'dict': {'hp': 13, 'sp': 5},
       'bool': [True, False, True, False]}


def test_caching(storage):
    # Write contents
    storage.write(doc)

    # Verify contents
    assert doc == storage.read()


def test_caching_read():
    db = TinyDB(storage=CachingMiddleware(MemoryStorage))
    assert db.all() == []


def test_caching_write_many(storage):
    storage.WRITE_CACHE_SIZE = 3

    # Storage should be still empty
    assert storage.memory is None

    # Write contents
    for x in range(2):
        storage.write(doc)
        assert storage.memory is None  # Still cached

    storage.write(doc)

    # Verify contents: Cache should be emptied and written to storage
    assert storage.memory


def test_caching_flush(storage):
    # Write contents
    for _ in range(CachingMiddleware.WRITE_CACHE_SIZE - 1):
        storage.write(doc)

    # Not yet flushed...
    assert storage.memory is None

    storage.write(doc)

    # Verify contents: Cache should be emptied and written to storage
    assert storage.memory


def test_caching_flush_manually(storage):
    # Write contents
    storage.write(doc)

    storage.flush()

    # Verify contents: Cache should be emptied and written to storage
    assert storage.memory


def test_caching_write(storage):
    # Write contents
    storage.write(doc)

    storage.close()

    # Verify contents: Cache should be emptied and written to storage
    assert storage.storage.memory


def test_nested():
    storage = CachingMiddleware(MemoryStorage)
    storage()  # Initialization

    # Write contents
    storage.write(doc)

    # Verify contents
    assert doc == storage.read()


def test_caching_json_write(tmpdir):
    path = str(tmpdir.join('test.db'))

    with TinyDB(path, storage=CachingMiddleware(JSONStorage)) as db:
        db.insert({'key': 'value'})

    # Verify database filesize
    statinfo = os.stat(path)
    assert statinfo.st_size != 0

    # Assert JSON file has been closed
    assert db._storage._handle.closed

    del db

    # Reopen database
    with TinyDB(path, storage=CachingMiddleware(JSONStorage)) as db:
        assert db.all() == [{'key': 'value'}]


def test_caching_rejects_use_after_close(tmpdir):
    path = str(tmpdir.join('closed.db'))
    db = TinyDB(path, storage=CachingMiddleware(JSONStorage))
    db.insert({'key': 'value'})
    db.close()

    with pytest.raises(ValueError, match='closed'):
        db.insert({'key': 'again'})

    with pytest.raises(ValueError, match='closed'):
        db.all()

    with TinyDB(path, storage=CachingMiddleware(JSONStorage)) as reopened:
        assert reopened.all() == [{'key': 'value'}]


def test_locking_read_write():
    db = TinyDB(storage=LockingMiddleware(MemoryStorage))
    db.insert({'key': 'value'})

    assert db.all() == [{'key': 'value'}]


def test_locking_nested_with_caching(tmpdir):
    path = str(tmpdir.join('locked.db'))

    with TinyDB(path,
                storage=LockingMiddleware(CachingMiddleware(JSONStorage))) as db:
        db.insert({'key': 'value'})

    with TinyDB(path) as db:
        assert db.all() == [{'key': 'value'}]


def test_locking_concurrent_readers_and_writer(tmpdir):
    # Without the lock, readers move the shared JSONStorage file cursor
    # while the writer is writing, which corrupts reads and the file itself
    path = str(tmpdir.join('locked.db'))
    db = TinyDB(path, storage=LockingMiddleware(JSONStorage))
    db.insert_multiple({'n': i} for i in range(100))

    errors = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            try:
                assert len(db.all()) >= 100
            except Exception as e:  # pragma: no cover
                errors.append(e)

    def writer():
        try:
            for i in range(100):
                db.insert({'n': 100 + i})
        except Exception as e:  # pragma: no cover
            errors.append(e)
        finally:
            # Always release the readers, or a failing writer hangs the test
            done.set()

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads.append(threading.Thread(target=writer))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(db) == 200
    db.close()

    with TinyDB(path) as reopened:
        assert len(reopened) == 200
