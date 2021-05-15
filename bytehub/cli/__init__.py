"""
The ByteHub command line interface (CLI) allows you to trigger execution of tasks. For example

    bytehub run test/my-task

will run `my-task` from within the `test` namespace.

Connection details for the feature store must be supplied as environment variables, e.g.

    BYTEHUB_CONNECTION=sqlite:///bytehub.db

for a SQLAlchemy connection, or

    BYTEHUB_TOKEN=eyJ...1VQ

where a refresh token must be used to connect to a cloud-hosted feature store.
"""
