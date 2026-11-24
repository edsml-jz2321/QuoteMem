# OpenSearch module

This folder is the complete built-in OpenSearch provider module. `meta.yaml`
publishes both `backend_type:opensearch` and its provider-local handler record
without importing the optional `opensearch-py` client during discovery.

The Backend role owns physical mappings, replacement documents, native lookup,
exhaustive scanning, and Query DSL execution. The Handler role pushes indexed
scalar operations and compatible text patterns into OpenSearch. Structural
types, null-sensitive filters, Python regular expressions, and wildcard forms
whose syntax differs from Python remain portable scan predicates.
