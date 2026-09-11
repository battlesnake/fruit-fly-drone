# Vertical T4/T5 commissioning manifest v1

This artifact freezes the exact existing T4c/T4d/T5c/T5d afferent-edge groups and the
training, development and acceptance stimulus specifications preregistered in commit
`461ca9d`.

`manifest.json` is intentionally a no-response artifact. Its generator does not render
pixels, load the neural checkpoint into a controller, or evaluate any neural activity.
In particular, the acceptance specifications are committed but their pixels remain sealed
until a development snapshot passes the preregistered gates.

The canonical manifest semantic SHA-256 is
`8d40ae087c1552f0473be47ede74a11f220c1d7474e4b4025a251839f1dcdfef`; the formatted
`manifest.json` file SHA-256 is
`1403c552b371378a59b9cd9830d06ce144f336c7adf05d5061fd385dfc5dbb88`. The generator file
SHA-256 at materialization is
`b8a39a9e324a7aad71e8e4e21aaf494c393019d2af13ad2fbfb4a280fdd57ea4`.

The artifact authorizes only implementing and running the disposable one-update preflight.
It authorizes no training run, retained candidate, hover test, gate test or promotion.
