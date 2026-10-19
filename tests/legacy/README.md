# Historical protocol tests

These unchanged tests predate the current settings and DAG-reference APIs. They
use removed fields such as `capability_traces`; they were already failing before
the branch separation. They remain available here for future migration and are
excluded from default discovery by `pytest.ini`.

Current regression coverage lives directly in `tests/`: protocol execution,
network accounting, dataset partitioning, literature baselines and numerical
checks for optimized execution. Run a historical file explicitly when porting it.
