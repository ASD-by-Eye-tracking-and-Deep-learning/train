# google-colab-cli manual

A walkthrough for driving Colab from the terminal instead of the browser
notebook UI — useful for jobs that outlive a laptop lid closing (like
`train_all.ipynb`'s CV runs). Source: https://github.com/googlecolab/google-colab-cli

Run these in order the first time. After that, skip straight to step 6/7
for day-to-day use.

## 1. Install

Already done on this machine (`google-colab-cli v0.7.4` via `uv tool`).
For reference / a fresh machine:

```bash
uv tool install google-colab-cli
colab version
```

## 2. Authenticate (one-time, interactive)

```bash
colab sessions
```

First call opens a browser OAuth consent screen (public InstalledAppFlow —
no `gcloud`/GCP setup needed). Approve it.

**Important**: sign in as the **"Duc n Huyen"** Google account specifically —
that's the account `ASD/ML4Autism/...` lives under on Drive. Signing in as a
different account means `colab drivemount` (step 4) mounts an empty/wrong
Drive later, and nothing will obviously tell you that's what happened.

Auth state is cached at `~/.config/colab-cli/sessions.json` — you won't be
asked again on this machine.

**Logged into the wrong Google account?** There's no `colab logout` command.
Force a fresh login instead:
```bash
rm ~/.config/colab-cli/token.json
colab sessions   # re-triggers the browser consent screen
```
If the browser auto-picks the same wrong account instead of showing a
chooser, sign out of that account in the browser first (or use an
incognito/private window) so "Duc n Huyen" is actually selectable.

## 3. Create a session

```bash
colab new -s myrun --gpu T4
colab sessions          # confirm it's listed
colab status -s myrun   # hardware, machine shape, current status
```

- `-s myrun` names the session — lets several coexist. If only one session
  is active, every other command below can drop `-s myrun`.
- `--gpu`: `T4`, `L4`, `G4`, `H100`, `A100` (availability depends on your
  Colab subscription tier — free tier is typically T4 only).
- `--tpu`: `v5e1` or `v6e1`, instead of `--gpu`.
- Omit both `--gpu`/`--tpu` for a CPU-only runtime.
- `--high-mem`: request a high-RAM shape (needs Colab Pro/Pro+ for most
  accelerators; ignored for L4/TPU, which only offer one shape).

## 4. Get files onto the VM

Two ways, use whichever fits:

**a) Upload individual files directly:**
```bash
colab upload -s myrun ./local_script.py remote_script.py
colab ls -s myrun
```

**b) Mount Google Drive** (better for this project — the dataset and code
are already mirrored to Drive via `train/sync_to_drive.sh`):
```bash
colab drivemount -s myrun
```
Mounts at `/content/drive` by default (same path a browser-based Colab
notebook uses) — everything under `MyDrive/ASD/ML4Autism/train/...` becomes
directly readable/writable from the VM, no per-file uploads needed.

## 5. Install Python dependencies

```bash
colab install -s myrun timm keras-tuner
```

`colab exec` has no notebook magics — `%pip install ...` doesn't apply here.
`colab install` is the CLI equivalent (uses `uv`, falls back to `pip`).
Colab VMs come with torch/torchvision/tensorflow/scikit-learn/scipy
preinstalled and matched to the GPU driver — don't reinstall those, just add
what's actually missing (like this project's notebook already does in its
own `%pip install` cell).

## 6. Run code

Three ways:

```bash
# a) Piped stdin — quick one-liners
echo 'print("hello from Colab")' | colab exec -s myrun

# b) An uploaded/local .py file (CLI reads it locally and sends the content —
#    no need to upload first)
colab exec -s myrun -f remote_script.py

# c) A .ipynb notebook — runs it against the VM kernel, writes outputs back
#    into the same file
colab exec -s myrun -f train_all.ipynb
```

Interactive alternatives if you want a live shell instead of one-shot exec:
```bash
colab repl -s myrun       # interactive Python REPL
colab console -s myrun    # raw TTY shell (tmux) on the VM
colab ssh -s myrun        # SSH over WebSocket (also works as an IDE ProxyCommand)
```

## 7. Check on a running job / pull results back

```bash
colab status -s myrun               # is it still running?
colab log -s myrun -n 50            # last 50 lines of session history
colab log -s myrun -o run.md        # export full log as Markdown/.ipynb/.jsonl/.txt
colab download -s myrun artifacts/some_result.json ./some_result.json
```

Sessions stay alive as long as the kernel is active (the Colab backend
handles liveness) — you don't need to keep a terminal open watching it, just
check back with `colab status`/`colab log` whenever.

## 8. Clean up

```bash
colab stop -s myrun
```

Releases the VM. Check usage/cost with:
```bash
colab usage    # compute-unit balance and burn rate
colab pay      # opens the Colab subscription page
```

## 9. One-shot ephemeral jobs (`colab run`)

For a single script that doesn't need a long-lived session: provisions a
fresh VM, runs the script (forwarding CLI args), retrieves output, tears the
VM down automatically — all in one command:

```bash
colab run --gpu T4 local_script.py --some-arg value
```

## Quick reference

| Command | Purpose |
|---|---|
| `colab new -s NAME [--gpu G] [--tpu T] [--high-mem]` | Create a session |
| `colab sessions` | List active sessions |
| `colab status -s NAME` | Hardware/status for a session |
| `colab restart-kernel -s NAME` | Restart the kernel |
| `colab stop -s NAME` | Terminate and release |
| `colab url -s NAME [--open]` | Get/open the browser URL for a session |
| `colab exec -s NAME [-f FILE]` | Run code (stdin, `.py`, or `.ipynb`) |
| `colab repl` / `colab console` / `colab ssh` | Interactive shells |
| `colab run [--gpu G] SCRIPT [ARGS]` | Ephemeral one-shot job |
| `colab upload/download/ls/rm/edit` | Remote file operations |
| `colab drivemount -s NAME [PATH]` | Mount Google Drive (default `/content/drive`) |
| `colab install -s NAME PKG...` | Install Python packages on the VM |
| `colab auth -s NAME` | Authenticate the VM for GCP services (BigQuery, GCS, etc.) |
| `colab log -s NAME [-n N] [-o FILE]` | View/export session history |
| `colab usage` / `colab pay` | Compute-unit balance / subscription page |

Global flags: `--auth {oauth2,adc}` (default `oauth2`), `-c/--client-oauth-config PATH`,
`--config PATH` (session metadata location), `--logtostderr`.
