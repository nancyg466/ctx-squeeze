# Nightly build postmortem

The nightly job started failing on Tuesday after the runner image was bumped.
Every run now spends eleven minutes reinstalling dependencies from scratch.

## Timeline

03:14 the scheduled job kicks off and pulls the latest runner image.
03:19 dependency installation begins from an empty cache.
03:31 the test suite starts against a cold environment.
03:42 the job finishes green but well past its usual six minute window.

## Investigation

Retry 1 hit the same failure: the runner image bump on Monday replaced the
base layer that carried the dependency cache, so the job rebuilds it from
scratch on every run and the install step times out after six minutes.

Retry 2 hit the same failure: the runner image bump on Monday replaced the
base layer that carried the dependency cache, so the job rebuilds it from
scratch on every run and the install step times out after six minutes.

```python
def restore_cache(job):
    job.mount_volume("dependency-cache")
    job.run("pip install -r requirements.txt --cache-dir dependency-cache")
```

## Fix

Restore the cache step and pin the runner image to the previous minor version.
Re-run the workflow twice to confirm the cache is populated and then read.

## Follow up

Add an alert that fires when the nightly job runs longer than eight minutes.
