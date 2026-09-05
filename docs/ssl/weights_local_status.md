# Local SSL 2-fold weights status

Checked when `ssl_sep` was wired (2026-09-05). Checkpoints are **not** in git (`checkpoints/` is ignored).

| Fold | File found locally | Epoch in file | Locked-test Dice of that file |
|---|---|---|---|
| 0 | `upload_packs/bhai-nnunet-fold0-ckpt/checkpoint_best.pth` | **510** | **0.4802** (verified) |
| 1 epoch **511** | **NOT FOUND** | — | **0.4965 is documented but the `.pth` is not on this machine** |
| 1 current pack | `upload_packs/bhai-nnunet-fold1-ckpt/checkpoint_best.pth` | **775** (EMA) | not the 0.4965 run |
| 1 older pack | `upload_26_08/.../checkpoint_best.pth` | **679** | **0.4418** (below 0.45) |

If `checkpoints/nnunet_ssl_2fold/fold_1/checkpoint_best.pth` is the 775 or 679 file, **do not** tell the UI that this ensemble is the 0.4965 model.

To restore the claim checkpoint: recover fold1 **epoch 511** `checkpoint_best.pth` from the Kaggle session that scored 0.4965, then replace `fold_1/checkpoint_best.pth`.
