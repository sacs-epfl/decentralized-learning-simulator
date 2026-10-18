"""Independent SGD tasks using bounded residency and accelerator batching.

Only the local executor uses this backend. DAG values and all aggregation remain
on CPU. CUDA is optional; the class also runs on CPU for numerical tests.
"""
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from dasklearn import functions
from dasklearn.datasets import create_dataset
from dasklearn.model_trainer import ModelTrainer
from dasklearn.models import create_model_structure, copy_model_state


def add_training_arguments(parser):
    parser.add_argument("--no-gpu-residency", dest="gpu_residency", action="store_false", default=True)
    parser.add_argument("--no-training-batching", dest="training_batching", action="store_false", default=True)
    parser.add_argument("--training-batch-size", type=int, default=2, help="Maximum independent models per training batch")
    parser.add_argument("--gpu-state-cache-mb", type=int, default=256)


class TrainingExecutor:
    def __init__(self, settings, autotune=True):
        self.settings = settings
        self.device = torch.device(settings.torch_device_name)
        self.width = settings.training_batch_size
        if self.width < 1 or settings.gpu_state_cache_mb < 0:
            raise ValueError("training_batch_size must be positive and gpu_state_cache_mb nonnegative")
        self.resident = settings.gpu_residency
        self.batch_enabled = settings.training_batching and self.width > 1
        self.autotune = autotune
        self.tuned = not autotune
        self.workspace = self.initial = self.bank = self.meta = None
        self.stream_models = []
        self.streams = []
        self.pool = None
        self.strategy = "cuda-streams" if self.device.type == "cuda" else "vmap"
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = settings.gpu_state_cache_mb * 1024 * 1024
        self.stats = {"device":str(self.device), "resident":self.resident,
                      "batching_requested":self.batch_enabled, "batched_tasks":0,
                      "serial_tasks":0, "batches":0, "calibration_tasks":0, "input_cache_hits":0,
                      "input_cache_peak_bytes":0, "fallback_reasons":[]}

    def note(self, reason):
        if reason not in self.stats["fallback_reasons"]:
            self.stats["fallback_reasons"].append(reason)

    def can_train(self, data):
        return (not self.settings.compute_validation_loss_global_model
                and not data.get("compute_gradient") and data.get("gradient_model") is None
                and data.get("local_steps", self.settings.learning.local_steps) > 0)

    def can_batch(self, data):
        # Native tensor datasets have deterministic __getitem__ and no random
        # augmentations. Other workloads keep the scalar task's RNG semantics.
        s = self.settings
        native = s.dataset in {"synthetic", "digits"} or (
            s.dataset_base_path and (s.dataset == "tiny_imagenet" and s.model == "resnet18"
                                    or s.dataset == "cifar10" and s.model in {"gnlenet", "lenet"}))
        return bool(native and self.can_train(data) and self.batch_enabled)

    def trainer(self, peer):
        if functions.dataset is None:
            functions.dataset = create_dataset(self.settings)
        if functions.model_trainers is None:
            functions.model_trainers = [None] * self.settings.participants
        if functions.model_trainers[peer] is None:
            functions.model_trainers[peer] = ModelTrainer(functions.dataset, self.settings, peer)
        return functions.model_trainers[peer]

    def input_model(self, data):
        model = data["model"]
        if isinstance(model, tuple):
            model = model[0]
        if model is None:
            if self.initial is None:
                self.initial = functions.initialize(self.settings, {})[0]
            return self.initial
        return model

    def drop_input(self, key):
        if key in self.cache:
            _, size = self.cache.pop(key)
            self.cache_bytes -= size

    def input_state(self, model, key):
        if key is not None and key in self.cache:
            self.cache.move_to_end(key)
            self.stats["input_cache_hits"] += 1
            return self.cache[key][0]
        state = model.state_dict()
        size = sum(t.numel() * t.element_size() for t in state.values())
        if self.resident and key is not None and size <= self.cache_limit:
            while self.cache and self.cache_bytes + size > self.cache_limit:
                self.drop_input(next(iter(self.cache)))
            state = {name:t.detach().to(self.device, copy=True) for name,t in state.items()}
            self.cache[key] = (state, size)
            self.cache_bytes += size
            self.stats["input_cache_peak_bytes"] = max(self.stats["input_cache_peak_bytes"], self.cache_bytes)
        return state

    def serial(self, data, key=None):
        if not self.resident or not self.can_train(data):
            return functions.train(self.settings, data)[0]
        model = self.input_model(data)
        if any(not p.requires_grad for p in model.parameters()):
            self.note("frozen parameters use reference training")
            return functions.train(self.settings, data)[0]
        if self.workspace is None:
            self.workspace = create_model_structure(self.settings.dataset, self.settings.model).to(self.device)
        self.workspace.load_state_dict(self.input_state(model, key))
        self.workspace.train()
        seed = functions.training_seed(self.settings, data)
        # Match functions.train: new optimizer/momentum per task, identical
        # data-loader generator and per-task seed. Never mutate the DAG input.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.trainer(data["peer"]).train(self.workspace,
                data.get("local_steps", self.settings.learning.local_steps),
                device_name=str(self.device), seed=seed, keep_on_device=True)
        output = copy_model_state(self.workspace, self.settings.dataset, self.settings.model)
        self.workspace.zero_grad(set_to_none=True)
        return output

    def batches(self, data):
        seed = functions.training_seed(self.settings, data)
        trainer = self.trainer(data["peer"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            loader = trainer.make_train_loader(seed)
            stream = iter(loader)
            result = []
            for _ in range(data.get("local_steps", self.settings.learning.local_steps)):
                try:
                    batch = next(stream)
                except StopIteration:
                    stream = iter(loader)
                    batch = next(stream)
                result.append(batch)
        return result

    def make_bank(self):
        model = create_model_structure(self.settings.dataset, self.settings.model).train()
        if any(isinstance(m, nn.modules.dropout._DropoutNd) and m.p for m in model.modules()):
            raise NotImplementedError("stochastic models require scalar RNG streams")
        if any(isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats and m.momentum is None
               for m in model.modules()):
            raise NotImplementedError("cumulative BatchNorm requires scalar counters")
        params = {name:torch.zeros((self.width, *p.shape), dtype=p.dtype, device=self.device, requires_grad=p.requires_grad)
                  for name,p in model.named_parameters()}
        buffers = {name:torch.zeros((self.width, *p.shape), dtype=p.dtype, device=self.device)
                   for name,p in model.named_buffers()}
        self.meta = model.to("meta")
        return params, buffers

    def vectorized(self, data, keys, batches):
        if self.bank is None:
            self.bank = self.make_bank()
        params, buffers = self.bank
        count = len(data)
        with torch.no_grad():
            for i,(item,key) in enumerate(zip(data,keys)):
                model = self.input_model(item)
                if any(not p.requires_grad for p in model.parameters()):
                    raise NotImplementedError("frozen parameters require scalar training")
                for name,value in self.input_state(model,key).items():
                    target = params[name] if name in params else buffers[name]
                    target[i].copy_(value)
            for p in params.values():
                p.grad = None
        s = self.settings
        # Sum independent per-model losses: averaging over models would change
        # each client's effective learning rate. SGD state is fresh per group.
        def loss(p,b,x,y):
            prediction = torch.func.functional_call(self.meta, (p,b), (x,), strict=True)
            if s.dataset == "cifar10" and s.model not in {"resnet8","resnet18","mobilenet_v3_large"}:
                return F.nll_loss(prediction,y)
            return F.cross_entropy(prediction,y)
        mapped_loss = torch.vmap(loss, randomness="error")
        optimizer = torch.optim.SGD(list(params.values()),lr=s.learning.learning_rate,
            momentum=s.learning.momentum,weight_decay=s.learning.weight_decay)
        feature = "image" if s.dataset == "tiny_imagenet" else "img"
        for step in range(len(batches[0])):
            pview,bview = ({k:v[:count] for k,v in group.items()} for group in (params,buffers))
            x = torch.stack([b[step][feature] for b in batches]).to(self.device)
            y = torch.stack([b[step]["label"] for b in batches]).to(self.device)
            optimizer.zero_grad(set_to_none=True)
            mapped_loss(pview,bview,x,y).sum().backward()
            optimizer.step()
        state = {name:t[:count].detach().cpu() for group in (params,buffers) for name,t in group.items()}
        outputs = []
        for i in range(count):
            model = create_model_structure(s.dataset,s.model)
            model.load_state_dict({name:t[i] for name,t in state.items()})
            outputs.append(model)
        for p in params.values():
            p.grad = None
        if not self.resident:
            self.bank = self.meta = None
        return outputs

    def concurrent_cuda(self, data, keys, batches):
        """Batch independent scalar kernels without changing their tensor shapes.

        Data/RNG preparation happens serially before launching workers. This is
        used only for the deterministic native model/data combinations allowed
        by can_batch, never for stochastic forwards or random augmentations.
        """
        count=len(data)
        while len(self.stream_models)<count:
            model=create_model_structure(self.settings.dataset,self.settings.model).to(self.device)
            if any(isinstance(m,nn.modules.dropout._DropoutNd) and m.p for m in model.modules()):
                raise NotImplementedError("stochastic forwards require scalar RNG streams")
            self.stream_models.append(model)
            self.streams.append(torch.cuda.Stream(device=self.device))
        if self.pool is None:
            self.pool=ThreadPoolExecutor(max_workers=self.width,thread_name_prefix="independent-sgd")
        s=self.settings
        optimizers=[]
        for model,item,key in zip(self.stream_models,data,keys):
            source=self.input_model(item)
            if any(not p.requires_grad for p in source.parameters()):
                raise NotImplementedError("frozen parameters require scalar training")
            model.load_state_dict(self.input_state(source,key));model.train()
            optimizers.append(torch.optim.SGD(model.parameters(),lr=s.learning.learning_rate,
                momentum=s.learning.momentum,weight_decay=s.learning.weight_decay))
        feature="image" if s.dataset=="tiny_imagenet" else "img"
        # Fuse transfers along the model axis while each worker still sees its
        # original contiguous minibatch. Partial epoch batches are preserved.
        xs=[torch.stack([b[step][feature] for b in batches]).to(self.device) for step in range(len(batches[0]))]
        ys=[torch.stack([b[step]["label"] for b in batches]).to(self.device) for step in range(len(batches[0]))]
        default=torch.cuda.current_stream(self.device)
        for stream in self.streams[:count]:stream.wait_stream(default)
        def worker(i):
            stream=self.streams[i];model=self.stream_models[i];optimizer=optimizers[i]
            with torch.cuda.device(self.device),torch.cuda.stream(stream):
                try:
                    for x,y in zip(xs,ys):
                        optimizer.zero_grad(set_to_none=True)
                        prediction=model(x[i])
                        loss=(F.nll_loss(prediction,y[i]) if s.dataset=="cifar10" and s.model not in {
                            "resnet8","resnet18","mobilenet_v3_large"} else F.cross_entropy(prediction,y[i]))
                        loss.backward();optimizer.step()
                    output=copy_model_state(model,s.dataset,s.model)
                    model.zero_grad(set_to_none=True)
                    return output
                finally:
                    # Also protect input/workspace lifetimes on an exception.
                    stream.synchronize()
        futures=[self.pool.submit(worker,i) for i in range(count)]
        try:
            return [future.result() for future in futures]
        finally:
            # Wait for every worker even if an earlier worker raised. Only then
            # may the default-stream input tensors and source cache be released.
            for future in futures:
                if not future.done():
                    try:future.result()
                    except Exception:pass
            for stream in self.streams[:count]:default.wait_stream(stream)
            if not self.resident:
                self.stream_models=[];self.streams=[]

    def batched(self, data, keys, batches):
        if self.strategy=="cuda-streams":return self.concurrent_cuda(data,keys,batches)
        return self.vectorized(data,keys,batches)

    def clear_batch_workspace(self):
        if self.pool:self.pool.shutdown(wait=True)
        self.pool=None;self.stream_models=[];self.streams=[]
        self.bank=self.meta=None

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run_group(self, data, keys, batches):
        if self.tuned:
            return self.batched(data,keys,batches)
        # Warm both paths, then compare complete task cost, including copies.
        for d,k in zip(data,keys):
            self.serial(d,k)
        self.batched(data,keys,batches)
        serial_times,batched_times=[],[]
        error=0.
        for _ in range(3):
            self.sync();start=time.perf_counter()
            reference = [self.serial(d,k) for d,k in zip(data,keys)]
            self.sync();serial_times.append(time.perf_counter()-start)
            start=time.perf_counter();candidate=self.batched(data,keys,batches)
            self.sync();batched_times.append(time.perf_counter()-start)
            for a,b in zip(reference,candidate):
                for name,value in a.state_dict().items():
                    other=b.state_dict()[name]
                    torch.testing.assert_close(value,other,rtol=1e-4,atol=1e-5)
                    if self.strategy=="cuda-streams" and not torch.equal(value,other):
                        raise AssertionError("CUDA stream batching must match scalar state exactly: "+name)
                    if value.is_floating_point():
                        error=max(error,float((value-other).abs().max()))
        serial_s=statistics.median(serial_times);batched_s=statistics.median(batched_times)
        self.stats["calibration"]={"serial_s":serial_s,"batched_s":batched_s,"max_state_abs_error":error,
                                   "serial_trials_s":serial_times,"batched_trials_s":batched_times}
        self.batch_enabled = batched_s < serial_s*.95
        self.tuned = True
        if not self.batch_enabled:
            self.note("batching was not faster in startup calibration")
            self.clear_batch_workspace()
        # Startup calibration itself always publishes the reference values.
        return reference

    def train_many(self, data, keys):
        if len(data)!=len(keys) or len(data)>self.width:
            raise ValueError("Supply matching task/key lists no longer than training_batch_size")
        outputs = [None]*len(data)
        groups=defaultdict(list)
        for i,item in enumerate(data):
            if self.can_batch(item):
                batch=self.batches(item)
                signature=tuple(tuple((k,tuple(v.shape),v.dtype) for k,v in sorted(b.items())) for b in batch)
                groups[signature].append((i,batch))
            else:
                outputs[i]=self.serial(item,keys[i]);self.stats["serial_tasks"]+=1
        for group in groups.values():
            indices=[i for i,_ in group]
            if len(group)>1 and self.batch_enabled:
                failure=None
                try:
                    calibrated = self.tuned
                    values=self.run_group([data[i] for i in indices],[keys[i] for i in indices],[b for _,b in group])
                except (NotImplementedError,AssertionError) as exc:
                    failure=type(exc).__name__+": "+str(exc)[:300]
                except RuntimeError as exc:
                    if not isinstance(exc,torch.OutOfMemoryError) and not any(word in str(exc) for word in ("vmap","BatchedTensor","batching rule","randomness")):
                        raise
                    failure=type(exc).__name__+": "+str(exc)[:300]
                if failure is None:
                    for i,value in zip(indices,values):outputs[i]=value
                    self.stats["batches"]+=int(calibrated)
                    self.stats["batched_tasks" if calibrated else "calibration_tasks"]+=len(group)
                    continue
                self.note(failure);self.batch_enabled=False;self.clear_batch_workspace()
                self.cache.clear();self.cache_bytes=0
                if self.device.type=="cuda":torch.cuda.empty_cache()
            for i in indices:
                outputs[i]=self.serial(data[i],keys[i]);self.stats["serial_tasks"]+=1
        return outputs

    def status(self):
        return {**self.stats,"batching_active":self.batch_enabled,"strategy":self.strategy,"input_cache_bytes":self.cache_bytes}

    def close(self):
        self.clear_batch_workspace()
        self.workspace=self.bank=self.meta=self.initial=None
        self.cache.clear();self.cache_bytes=0
