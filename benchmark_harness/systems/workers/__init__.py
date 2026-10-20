"""Disposable, one-task workers for the pinned non-ranking adapters.

Workers are deliberately launched by the host for one task only.  They have
no provider credentials, no network client, and communicate through the v3
JSONL channel exclusively.
"""
