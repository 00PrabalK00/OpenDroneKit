# Credentials

Every credential this project uses is read from the environment and never written to a
file the repository tracks. That is a property worth checking rather than trusting, so
each entry below says how to verify it.

## What the project reads

| Variable | Used by | Needed for |
|---|---|---|
| `ROBOFLOW_API_KEY` | `training/datasets/download.py`, `training/train_det.py`, `training/cloud/vast_bootstrap.sh` | Downloading the detection corpora. Unset means those datasets are skipped, with a message saying so — never a silent substitution. |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` (or `~/.kaggle/kaggle.json`) | `tools/kaggle_kernel.py` | Pushing training kernels. |

`.gitignore` covers `.env`, and no credential is ever passed on a command line, where it
would land in shell history and in the process table.

## Verifying nothing leaked

Run these before publishing. Both should print nothing.

```bash
# Is the secret in any tracked file right now?
git grep -In "<the secret>"

# Was it ever in any commit, on any branch?
git log --all -S"<the secret>" --oneline
```

The second matters more than the first. Deleting a secret from a file does not remove it
from history, and a repository that looks clean at HEAD can still be carrying it.

If either command prints anything, rotating the key is not optional and rewriting history
is not sufficient on its own — assume the old value is compromised and revoke it.

## Rotating the Roboflow key

Do this yourself; the steps involve signing in, and nothing here should be handed to an
agent.

1. Sign in to Roboflow, open **Settings → API Keys**, and issue a new private key.
2. **Revoke the old key** in the same screen. Issuing a replacement does not disable the
   previous one — an un-revoked old key is still a live credential.
3. Set the new value in your shell profile or `.env`:

   ```bash
   export ROBOFLOW_API_KEY=<new value>
   ```

4. Confirm the project picks it up:

   ```bash
   python -m training.datasets.download --status   # prints "roboflow : yes"
   ```

5. Update it anywhere else it is stored — the Kaggle kernel's secrets and any Vast.ai
   instance environment, which `training/cloud/vast_bootstrap.sh` reads.

Rotate whenever a key has been pasted into a chat transcript, a screenshot, an issue, or
a CI log, whether or not it reached a tracked file. Those are all places it can be read
from later.
