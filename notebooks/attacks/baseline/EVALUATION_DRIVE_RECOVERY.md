# Evaluation Drive recovery

The final GPT-5 mini evaluation worker exited with `TimeoutError: The read operation timed out` inside google-auth-httplib2 while saving a checkpoint to Drive. Its 894 successful captions and 28 successful unique judge responses remained in local checkpoints. Gemini was pending because its API key was intentionally deferred.

All three open VLMs already had complete summaries: 200 judged pairs per attack, 600 per VLM. Unique judge-response counts can be smaller because identical caption pairs reuse the same response.

`molab_evaluation_retry_cell.py` installs the opt-in `transport_support/sitecustomize.py` hook outside the frozen source bundle. It places the hook and run root in the child `PYTHONPATH`, enables `BASELINE_DRIVE_TRANSPORT_RETRY`, and calls the existing idempotent launcher. No original attack code, evaluation protocol, frozen worker, or experiment fingerprint is changed.

The hook wraps only Drive listing, upload and download operations. It retries identified transport exceptions at most four attempts, with 5/10/20 second backoff, recreating the authenticated HTTP connection with a 180 second timeout. It does not retry checksum/identity validation errors or provider HTTP/API errors. Existing MD5 and Drive-ID handling remains in the original storage methods.

Verification: simulated transient timeouts recover, persistent timeouts stop after four attempts, and validation errors propagate immediately. The live resumed GPT-5 mini unique judge-response count advanced from 28 to 36 with zero checkpointed evaluation errors. Existing captions were reused.
