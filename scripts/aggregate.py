import time

from dasklearn.model_manager import ModelManager
from dasklearn.models import create_model
from dasklearn.session_settings import SessionSettings, LearningSettings

print("Creating models...")
models = [create_model("femnist") for _ in range(20)]

settings = SessionSettings(
    algorithm="dpsgd",
    seed=42,
    dataset="femnist",
    learning=LearningSettings(batch_size=0, learning_rate=0, momentum=0, weight_decay=0, local_steps=0),
    participants=100
)

model_manager = ModelManager(None, settings, 0)
for ind, model in enumerate(models):
    model_manager.process_incoming_trained_model("%d" % ind, model)

start_time = time.time()
agg_model = model_manager.aggregate_trained_models()
print("Model aggregation took %f s." % (time.time() - start_time))
