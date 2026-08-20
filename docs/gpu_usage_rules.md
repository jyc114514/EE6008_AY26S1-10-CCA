# GPU Server usage rules

The server workspace is shared storage, not a second unstructured copy of the Git repository.

| Area | Rule |
| --- | --- |
| data_raw/ | Read-only shared dataset storage. Do not duplicate or modify raw files in a personal output directory. |
| repo/ | Clone/sync the GitHub repository. Keep code and configs under version control. |
| checkpoints/ | Shared model checkpoints; record the exact identifier/path in the run ledger, never commit the file. |
| features/ | Large extracted tensors; server-only. |
| outputs/<member>/<run_id>/ | Each member owns an isolated run directory. Do not use a shared outputs/current/. |
| logs/ and tmp/ | Runtime artifacts and temporary files; clean up after the retention period agreed by the team. |
| results/ | Small, curated summaries only. Copy a result into GitHub only after checking size, provenance, and privacy. |

Every run should use outputs/<member>/<run_id>/ and have a matching row in experiments/runs.csv. Do not place generated outputs directly under the repository root.

## Security and storage

Never put passwords, tokens, API keys, SSH keys, private forms, credentials, or personal identifiers in the repository or shared logs. Keep raw data and model weights on approved server storage. Check file sizes before copying anything to GitHub.
