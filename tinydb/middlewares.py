"""
Contains the :class:`base class <tinydb.middlewares.Middleware>` for
middlewares and implementations.
"""
import threading
from typing import Optional

from tinydb import Storage


class Middleware:
    """
    The base class for all Middlewares.

    Middlewares hook into the read/write process of TinyDB allowing you to
    extend the behaviour by adding caching, logging, ...

    Your middleware's ``__init__`` method has to call the parent class
    constructor so the middleware chain can be configured properly.
    """

    def __init__(self, storage_cls) -> None:
        self._storage_cls = storage_cls
        self.storage: Storage = None  # type: ignore

    def __call__(self, *args, **kwargs):
        """
        Create the storage instance and store it as self.storage.

        Usually a user creates a new TinyDB instance like this::

            TinyDB(storage=StorageClass)

        The storage keyword argument is used by TinyDB this way::

            self.storage = storage(*args, **kwargs)

        As we can see, ``storage(...)`` runs the constructor and returns the
        new storage instance.


        Using Middlewares, the user will call::

                                       The 'real' storage class
                                       v
            TinyDB(storage=Middleware(StorageClass))
                       ^
                       Already an instance!

        So, when running ``self.storage = storage(*args, **kwargs)`` Python
        now will call ``__call__`` and TinyDB will expect the return value to
        be the storage (or Middleware) instance. Returning the instance is
        simple, but we also got the underlying (*real*) StorageClass as an
        __init__ argument that still is not an instance.
        So, we initialize it in __call__ forwarding any arguments we receive
        from TinyDB (``TinyDB(arg1, kwarg1=value, storage=...)``).

        In case of nested Middlewares, calling the instance as if it was a
        class results in calling ``__call__`` what initializes the next
        nested Middleware that itself will initialize the next Middleware and
        so on.
        """

        self.storage = self._storage_cls(*args, **kwargs)

        return self

    def __getattr__(self, name):
        """
        Forward all unknown attribute calls to the underlying storage, so we
        remain as transparent as possible.
        """

        return getattr(self.__dict__['storage'], name)


class CachingMiddleware(Middleware):
    """
    Add some caching to TinyDB.

    This Middleware aims to improve the performance of TinyDB by writing only
    the last DB state every :attr:`WRITE_CACHE_SIZE` time and reading always
    from cache.
    """

    #: The number of write operations to cache before writing to disc
    WRITE_CACHE_SIZE = 1000

    def __init__(self, storage_cls):
        # Initialize the parent constructor
        super().__init__(storage_cls)

        # Prepare the cache
        self.cache = None
        self._cache_modified_count = 0
        self._closed = False

    def read(self):
        if self._closed:
            raise ValueError('I/O operation on closed storage')

        if self.cache is None:
            # Empty cache: read from the storage
            self.cache = self.storage.read()

        # Return the cached data
        return self.cache

    def write(self, data):
        if self._closed:
            raise ValueError('I/O operation on closed storage')

        # Store data in cache
        self.cache = data
        self._cache_modified_count += 1

        # Check if we need to flush the cache
        if self._cache_modified_count >= self.WRITE_CACHE_SIZE:
            self.flush()

    def flush(self):
        """
        Flush all unwritten data to disk.
        """
        if self._closed:
            raise ValueError('I/O operation on closed storage')

        if self._cache_modified_count > 0:
            # Force-flush the cache by writing the data to the storage
            self.storage.write(self.cache)
            self._cache_modified_count = 0

    def close(self):
        if self._closed:
            return

        # Flush potentially unwritten data
        self.flush()

        # Let the storage clean up too
        self.storage.close()
        self._closed = True
        self.cache = None


class LockingMiddleware(Middleware):
    """
    Make a storage safe to share between threads.

    TinyDB itself does no locking. When several threads use the same TinyDB
    instance, their reads and writes interleave on the same storage, and
    with :class:`~tinydb.storages.JSONStorage` they even share a single file
    handle and cursor, which corrupts the database file.

    This middleware serializes all access to the wrapped storage::

        db = TinyDB('db.json', storage=LockingMiddleware(JSONStorage))

    It can be combined with other middlewares. Put it outermost so the lock
    also covers them::

        storage = LockingMiddleware(CachingMiddleware(JSONStorage))

    Note that this does not make TinyDB safe for use from multiple
    *processes*.
    """

    def __init__(self, storage_cls):
        # Initialize the parent constructor
        super().__init__(storage_cls)

        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            return self.storage.read()

    def write(self, data):
        with self._lock:
            self.storage.write(data)

    def close(self):
        with self._lock:
            self.storage.close()
