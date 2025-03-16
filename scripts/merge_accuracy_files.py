import json
import os
import sys


def merge_accuracy_files(data_dir: str):
    # Read settings file
    settings_path = os.path.join(data_dir, "settings.json")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings file {settings_path} does not exist.")
    
    with open(settings_path, "r") as settings_file:
        settings = json.load(settings_file)

    paths = [os.path.join(data_dir, "accuracies_" + str(i) + ".csv") for i in range(settings["participants"])]

    with open(os.path.join(data_dir, "accuracies.csv"), "w") as output_file:
        output_file.write("algorithm,dataset,partitioner,alpha,peer,round,time,accuracy,loss\n")
        for path in paths:
            if os.path.exists(path):
                with open(path, "r") as input_file:
                    next(input_file)  # Ignore the header
                    output_file.write(input_file.read())
                os.remove(path)
            else:
                print(f"File {path} not found.")

    print("Merged accuracy files.")


if __name__ == "__main__":
    merge_accuracy_files(sys.argv[1])
