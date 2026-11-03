# Historical Molab implementation

`legacy_v1/` preserves the previous pipeline/recovery code, configuration, surrogate dependencies and synthetic tests. `PRESERVED_FILES.json` records pre-migration byte hashes. Original experiment ZIPs and Drive snapshots remain unchanged.

The editable implementation now lives in the four MPC-named scripts at the parent directory. Those scripts do not import this legacy package.

Historical launch instructions and source locks refer to the original packaged layout. Reproduce historical runs from their original verified source ZIP, rather than treating this moved folder as a new approved launch package.

Offline historical checks:

```bash
python -X utf8 -m unittest discover -s src/My-Attack-V3/molab/legacy_v1 -p "test_*.py" -v
```
