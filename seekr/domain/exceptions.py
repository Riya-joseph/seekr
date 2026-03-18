"""Domain-level exceptions for Seekr."""


class SeekriError(Exception):
    """Base exception for all Seekr errors."""


class IndexingError(SeekriError):
    """Raised when indexing a file fails."""


class SearchError(SeekriError):
    """Raised when a search operation fails."""


class ParseError(SeekriError):
    """Raised when a file cannot be parsed."""


class StoreError(SeekriError):
    """Raised when a vector store operation fails."""


class ModelError(SeekriError):
    """Raised when an embedding model operation fails."""


class WatcherError(SeekriError):
    """Raised when a file watcher operation fails."""
