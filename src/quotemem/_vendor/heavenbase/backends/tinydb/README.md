# TinyDB module

This folder is the complete built-in TinyDB provider module. `meta.yaml`
publishes the `backend_type:tinydb` record while the optional `tinydb` package
is imported only when a backend instance opens its database.

The Backend role uses one canonical local JSON file per workspace and backend
instance. It implements replacement semantics explicitly because TinyDB's
native upsert operation merges documents. TinyDB has no physical scalar index
facility, so the module does not claim scalar-index support.
