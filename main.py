from multiprocessing import freeze_support

from dask.distributed import Client

from dasklearn.model_manager import ModelManager
from dasklearn.models import create_model
from dasklearn.session_settings import SessionSettings, LearningSettings

if __name__ == "__main__":
    freeze_support()

    learning_settings = LearningSettings(
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0,
        batch_size=20,
        local_steps=5,
    )

    settings = SessionSettings(
        dataset="cifar10",
        work_dir="",
        learning=learning_settings,
        participants=1,
        target_participants=1,
        partitioner="iid",
    )

    # Start a local Dask cluster and connect to it
    client = Client()
    print("Client URL dashboard: %s" % client.dashboard_link)

    def train(params):
        model, round_nr = params

        model_manager = ModelManager(model, settings, 0)
        model_manager.train()

        print("Training in round %d..." % round_nr)
        return model

    # Create an initial model
    initial_model = create_model("cifar10")
    dsk = {
        'r0': initial_model,
    }
    for r in range(1, 20):
        dsk['r%d' % r] = (train, ['r%d' % (r - 1), r])

    # Submit the tasks
    print("Starting training...")
    result = client.get(dsk, 'r19')

    print(result)

    while True:
        pass
