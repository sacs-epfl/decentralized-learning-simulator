"""Causal, parameter-light placement heuristic; no simulator capacity access.

The score is a port-load lower bound plus a serialization/pipeline tail proxy,
not an optimality guarantee. Fixed owners retain all optimizer state. Geometry
is frozen after profiling; feedback subsequently changes dispatch priorities.
"""
from collections import deque
from math import ceil, log2
from statistics import median

HISTORY = 5
DISPATCH_US = 50          # controlled CPU assumption, not a device measurement
CANDIDATE_US = 100       # likewise; emitted alongside measured planner wall time


def pairings(nodes, seed):
    """O(log N) disjoint matching probes, reproducible and topology agnostic."""
    from random import Random
    order = list(range(nodes))
    Random(seed ^ 0xC0F103).shuffle(order)
    if nodes % 2:
        order.append(None)
    result = []
    for _ in range(min(len(order)-1, max(1, ceil(log2(nodes))))):
        result.append([(a, b) for a, b in zip(order[:len(order)//2], reversed(order[len(order)//2:]))
                       if a is not None and b is not None])
        order = [order[0], order[-1], *order[1:-1]]
    return result


def summarize_probes(records, training, overhead):
    """Two sizes cancel fixed latency; max observed peer rate estimates ports.

    Limited matchings can underestimate a fast endpoint whose partners are all
    slow. This is intentionally an estimate, not a peek at the simulator oracle.
    """
    grouped = {}
    for r in records:
        grouped.setdefault((r['src'], r['dst'], r['phase']//2), {})[r['phase'] % 2] = r
    incoming = {i: [] for i in training}; outgoing = {i: [] for i in training}
    latencies = {i: [] for i in training}
    for (src, dst, _), sizes in grouped.items():
        if len(sizes) != 2:
            raise ValueError('Incomplete two-size probe')
        a, b = sizes[0], sizes[1]
        dt = b['elapsed'] - a['elapsed']
        rate = (b['size']-a['size'])*1e6/dt if dt > 0 else a['size']*1e6/max(1, a['elapsed'])
        rate = max(1, round(rate))
        outgoing[src].append(rate); incoming[dst].append(rate)
        latency = max(0, a['elapsed'] - (a['size']+overhead)*1e6/rate)
        latencies[src].append(latency); latencies[dst].append(latency)
    return {i: (max(outgoing[i]), max(incoming[i]), training[i], round(median(latencies[i])))
            for i in sorted(training)}


def allocate(count, eligible, profiles, model_bytes, sample_size):
    """Discrete water filling including expected trainer/recipient port loads."""
    n = len(profiles); q = sample_size/n; loads = dict.fromkeys(eligible, 0)
    for _ in range(count):
        def score(i):
            x = (loads[i]+1)*model_bytes/count
            up, down, _, _ = profiles[i]
            traffic = q*model_bytes + (sample_size-2*q)*x
            return max(traffic/up, traffic/down), loads[i], i
        loads[min(eligible, key=score)] += 1
    # Interleave owners rather than putting every range at one owner first.
    remaining = dict(loads); owners = []
    for c in range(count):
        i = max(eligible, key=lambda p: (loads[p]*(c+1)/count-(loads[p]-remaining[p])
                                       if remaining[p] else -float('inf'), -p))
        owners.append(i); remaining[i] -= 1
    return owners


def score_layout(owners, profiles, samples, model_bytes, overhead, optimizer_us):
    """Account for local deliveries and changing source/destination overlaps."""
    n = len(profiles); count = len(owners); b = model_bytes/count
    loads = {i: owners.count(i) for i in profiles}
    latency = max(p[3] for p in profiles.values())/1e6
    values = []
    for src, dst in samples:
        src, dst = set(src), set(dst); m, d = len(src), len(dst)
        lower = 0.; tail = 0.
        for i, (up, down, train, _) in profiles.items():
            x = loads[i]*b; s = int(i in src); z = int(i in dst)
            outgoing = s*(model_bytes-x)+(d-z)*x
            incoming = (m-s)*x+z*(model_bytes-x)
            out_messages = s*(count-loads[i])+(d-z)*loads[i]
            in_messages = (m-s)*loads[i]+z*(count-loads[i])
            lower = max(lower, (outgoing+out_messages*overhead)/up,
                        (incoming+in_messages*overhead)/down, loads[i]*optimizer_us/1e6)
            if s:
                lower = max(lower, train/1e6+(model_bytes-x+(count-loads[i])*overhead)/up)
            if loads[i]:
                tail = max(tail, (b+overhead)*((m-s)/down+(d-z)/up)+optimizer_us/1e6)
        # Streaming may overlap port work; tail proxy is conservative, not a
        # full event simulation. All candidate layouts use the same formula.
        values.append(lower+tail+2*latency)
    return sum(values)/len(values)


def choose_layout(profiles, sample, model_bytes, elements, overhead, optimizer_us):
    """Geometric geometry/worker search, independent of accuracy and test seeds."""
    n = len(profiles); m = len(sample(1))
    samples = [(sample(r), sample(r+1)) for r in range(1, ceil(n/m)+1)]
    ranked = sorted(profiles, key=lambda i: (-min(profiles[i][:2]), i))
    # Header-dominated partitions cannot improve this model. The last candidate
    # is included even when not a power of two; never create empty tensor slices.
    max_count = max(1, min(elements, ceil(model_bytes/max(1, overhead))))
    counts = sorted({1, max_count, *[2**k for k in range(ceil(log2(max_count))+1) if 2**k <= max_count]})
    widths = sorted({n, *[2**k for k in range(ceil(log2(n))+1) if 2**k <= n]})
    candidates = []
    for count in counts:
        for width in widths:
            if width > count:
                continue
            owners = allocate(count, ranked[:width], profiles, model_bytes, m)
            cost = score_layout(owners, profiles, samples, model_bytes, overhead, optimizer_us)
            candidates.append(dict(chunks=count, reducers=len(set(owners)), predicted_seconds=cost, owners=owners))
    # Screen geometry with the cheap fluid proxy, then resolve one geometry per
    # worker width using the measured, numerical-free transport forecast. This
    # captures collection/dissemination dependencies absent from a port bound.
    from .forecast import forecast_score
    finalists=[]
    for width in sorted({c['reducers'] for c in candidates}):
        group=[c for c in candidates if c['reducers']==width]
        if group:
            finalists.append(min(group,key=lambda c:(c['predicted_seconds'],c['chunks'])))
    for c in finalists:
        c['forecast_seconds']=forecast_score(c['owners'],profiles,samples,model_bytes,overhead,optimizer_us)
    best=min(finalists,key=lambda c:(c['forecast_seconds'],c['chunks'],c['reducers']))
    return best, [{k: v for k, v in c.items() if k != 'owners'} for c in candidates]


class RouteFeedback:
    """Observed route delay, explicitly NOT an endpoint-bandwidth estimator."""
    def __init__(self):
        self.history = {}

    def observe(self, peer, actual_us, expected_us):
        self.history.setdefault(peer, deque(maxlen=HISTORY)).append(actual_us/max(1, expected_us))

    def factor(self, peer):
        values = self.history.get(peer)
        return max(1., median(values)) if values else 1.
