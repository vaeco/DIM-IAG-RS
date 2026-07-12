# Minimal Server Deployment

The target project path is:

```text
/root/work/DIM_WRR_IAG_RS_RELEASE
```

The clean source archive does not contain checkpoints. Before replacing the old folder, preserve the conservative checkpoint and copy the retrained bright checkpoint into the new `weights/` directory.

## 1. Backup the current Release

```bash
cd /root/work
STAMP=$(date +%Y%m%d_%H%M%S)
OLD_RELEASE=/root/work/DIM_WRR_IAG_RS_RELEASE
BACKUP=/root/work/DIM_WRR_IAG_RS_RELEASE_backup_${STAMP}

mv "$OLD_RELEASE" "$BACKUP"
echo "BACKUP=$BACKUP"
```

## 2. Extract the clean Release

Upload `DIM-IAG-RS_FINAL_RELEASE.zip` to `/root/work/`, then run:

```bash
cd /root/work
rm -rf /root/work/_dim_iag_rs_release_unpack
mkdir -p /root/work/_dim_iag_rs_release_unpack

unzip -q \
/root/work/DIM-IAG-RS_FINAL_RELEASE.zip \
-d /root/work/_dim_iag_rs_release_unpack

mv \
/root/work/_dim_iag_rs_release_unpack/DIM_WRR_IAG_RS_RELEASE \
/root/work/DIM_WRR_IAG_RS_RELEASE
```

## 3. Restore the two checkpoints

Use the `BACKUP` value printed in step 1:

```bash
mkdir -p \
/root/work/DIM_WRR_IAG_RS_RELEASE/weights

cp \
"$BACKUP/weights/dim_iag_rs_best.pth" \
/root/work/DIM_WRR_IAG_RS_RELEASE/weights/dim_iag_rs_best.pth

cp \
/root/work/snapshot_lol_iag_rs_bright_gate/best.pth \
/root/work/DIM_WRR_IAG_RS_RELEASE/weights/dim_iag_rs_bright_gate_best.pth
```

## 4. Check the Release

```bash
cd /root/work/DIM_WRR_IAG_RS_RELEASE
conda activate base

python -m py_compile \
scripts/enhance_dim_iag_rs.py

python scripts/validate_release.py
```

The final line should be:

```text
========== RELEASE CHECK PASSED ==========
```

## 5. One-image smoke test

```bash
cd /root/work/DIM_WRR_IAG_RS_RELEASE

rm -rf /root/work/debug_dual_gate/release_smoke

python -u scripts/enhance_dim_iag_rs.py \
  --input_dir /root/work/datasets/CODaN/test_night \
  --output_dir /root/work/debug_dual_gate/release_smoke \
  --max_images 1 \
  --balanced_subdirs \
  --print_freq 1
```

The log should end with:

```text
========== FINAL STATS ==========
Done.
```
