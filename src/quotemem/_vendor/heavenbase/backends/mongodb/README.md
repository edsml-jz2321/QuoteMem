# MongoDB module

This folder is the complete built-in MongoDB provider module. `meta.yaml`
publishes the `backend_type:mongodb` record without importing the optional
`pymongo` driver; runtime imports occur only when a MongoDB backend is
constructed.

The Backend role stores replacement row fragments as native MongoDB documents.
A bound workspace owns a normalized database name, and each backend instance
adds its name to collection names so multiple MongoDB instances cannot collide.
Scalar index declarations are reconciled as provider indexes during schema
setup.
