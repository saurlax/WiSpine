# Data Directory

This directory is reserved for the WiSpine CSI dataset. The raw data files are
not tracked in git because they are large and may need to be downloaded
separately.

After downloading the dataset, place the three class folders directly under
this directory:

```text
data/
├── kyphosis/
│   ├── cyd-k01.dat
│   ├── ...
│   └── zzy-k050.dat
├── normal/
│   ├── cyd-n01.dat
│   ├── ...
│   └── zzy-n050.dat
└── scoliosis/
    ├── cyd-s01.dat
    ├── ...
    └── zzy-s050.dat
```

Expected folders:

- `kyphosis/`: CSI samples for kyphosis/postural hunchback cases.
- `normal/`: CSI samples for normal posture cases.
- `scoliosis/`: CSI samples for scoliosis cases.

Each folder is expected to contain `.dat` files. File names follow this pattern:

```text
<subject>-<class><sample>.dat
```

Examples:

- `cyd-k01.dat`: subject `cyd`, kyphosis sample 01.
- `cyd-n01.dat`: subject `cyd`, normal sample 01.
- `cyd-s01.dat`: subject `cyd`, scoliosis sample 01.

The scripts assume this layout. For example:

```powershell
uv run python scripts/inspect_csi.py data/kyphosis/cyd-k01.dat
```
