import os
import h5py

ZENODO_DIR = "/home/thibeau.wouters/gw-datasets/GWTC-3/"
all_zenodo_files = [f for f in os.listdir(ZENODO_DIR) if f.endswith(".h5")]
example_file = os.path.join(ZENODO_DIR, "IGWN-GWTC3p0-v2-GW191204_171526_PEDataRelease_mixed_nocosmo.h5")

example_event_id = "GW191204_171526"

def my_decode(x):
    return x[0].decode('utf-8')

def load_event_metadata(event_id: str,
                        print_config: bool = True):
    
    # Find the correct filename
    filename_list = [f for f in all_zenodo_files if event_id in f and "nocosmo" in f]
    assert len(filename_list) == 1, f"Found {len(filename_list)} files for event {event_id}, something went wrong."
    
    with h5py.File(os.path.join(ZENODO_DIR, filename_list[0]), "r") as f:
        print(f.keys())
        
        # Get the config
        data = f["C01:IMRPhenomXPHM"]
        config = data["config_file"]["config"]
        if print_config:
            for key, value in config.items():
                print(f"{key}: {value[()]}")
        
        metadata = {}
        metadata["outdir"] = str(my_decode(config["outdir"][()]))
        metadata["duration"] = float(config["duration"][()])
        metadata["detectors"] = eval(my_decode(config["detectors"][()])) # eval: list to string of lists
        
        print(metadata)
        
# Test it
load_event_metadata(example_event_id)