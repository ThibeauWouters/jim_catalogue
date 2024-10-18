# TODO: also save the PE samples for plotting?
# TODO: make plots of the data just to be sure?

import os
import copy
import json
import pickle
import h5py
import numpy as np
import pandas as pd

################
### PREAMBLE ###
################

df = pd.read_csv("./GWTC3_confident.csv")
event_ids = df["commonName"].values

ZENODO_DIR = "/home/thibeau.wouters/gw-datasets/GWTC-3/"
all_zenodo_files = [f for f in os.listdir(ZENODO_DIR) if "nocosmo" in f and f.endswith(".h5")]

print("all_zenodo_files")
print(all_zenodo_files)

def my_decode(x):
    return x[0].decode('utf-8')

def load_event_metadata(event_id: str,
                        print_config: bool = True) -> dict:
    """
    Loads the event metadata from a Zenodo HDF5 file.

    Args:
        event_id (str): Event ID, such as "GW191204_171526".
        print_config (bool, optional): Whether to print the entire content of the HDF5 file to inspect available options. Defaults to True.
    """
    
    # Find the correct filename
    filename_list = [f for f in all_zenodo_files if event_id in f and "nocosmo" in f]
    if len(filename_list) != 1:
        print(f"Found {len(filename_list)} files for event {event_id}, something went wrong.")
        print("Skipping")
        
        return {}
    
    with h5py.File(os.path.join(ZENODO_DIR, filename_list[0]), "r") as f:
        print(f.keys())
        
        # Get the config
        try:
            data = f["C01:IMRPhenomXPHM"]
        except Exception as e:
            print(f"Error when trying to get the data: {e}")
            print("Trying with fetching HighSPin instead")
            data = f["C01:IMRPhenomXPHM:HighSpin"]
            
        config = data["config_file"]["config"]
        if print_config:
            print("--------------------------------")
            for key, value in config.items():
                print(f"{key}: {value[()]}")
            print("--------------------------------")
        
        # Get metadata
        metadata = {}
        metadata["outdir"] = str(my_decode(config["outdir"][()]))
        metadata["duration"] = float(my_decode(config["duration"][()]))
        metadata["detectors"] = eval(my_decode(config["detectors"][()])) 
        metadata["trigger_time"] = eval(my_decode(config["trigger-time"][()]))
        try:
            metadata["f_min"] = eval(my_decode(config["minimum-frequency"][()]))
            metadata["f_max"] = eval(my_decode(config["maximum-frequency"][()]))
        except Exception as e:
            print(f"Error when trying to get f_min and f_max: {e}")
        
        # TODO: not sure if channels are needed?
        # metadata["channel_dict"] = eval(my_decode(config["channel-dict"][()]))
        
    return metadata

def get_data_and_psd(run_dir: str, outdir: str = "./outdir/") -> int:
    """
    Get the data and psd values from the outdir
    
    Args:
        run_dir (str): The run directory of the injection
        outdir (str): The outdir of the injection where we save the data for Jim consumption
    """
    
    if run_dir.startswith("./"):
        print("run_dir starts with ./, this is not allowed")
        return 0
    
    try:
        data_dir = os.path.join(run_dir, "data")
        datadump_file_list = [f for f in os.listdir(data_dir) if "data_dump" in f and (f.endswith("pkl") or f.endswith("pickle"))]
        assert len(datadump_file_list) == 1, f"Found {len(datadump_file_list)} files in {data_dir}, something went wrong."
        
        datadump_file = os.path.join(data_dir, datadump_file_list[0])
        with open(datadump_file, "rb") as f:
            data = pickle.load(f)
            
            try:
                ifo_list = data.interferometers
            
            except Exception as e:
                print(f"Error when trying data.interferometers: {e}")
                print(f"Trying with dict method instead")
                ifo_list = data["ifo_list"]
                
    except Exception as e:
        print(f"Error when trying to open the datadump file: {e}")
        return
        
    for ifo in ifo_list:
        # Get the strain
        strain_data = ifo.strain_data # InterferometerStrainData
        _frequency_domain_strain = strain_data._frequency_domain_strain
        _times_and_frequencies = strain_data._times_and_frequencies # CoupledTimeAndFrequencySeries object
        
        frequencies = _times_and_frequencies.frequency_array
        real_strain = _frequency_domain_strain.real
        imag_strain = _frequency_domain_strain.imag
        
        print("np.max(frequencies)")
        print(np.max(frequencies))
        
        # Get the PSD values
        psd_values = ifo.power_spectral_density._PowerSpectralDensity__power_spectral_density_interpolated(frequencies)
        
        # Assert all have same length
        assert len(frequencies) == len(real_strain) == len(imag_strain) == len(psd_values), "Some shape mismatch"
        
        # Assert no NaNs or infs
        assert not np.any(np.isnan(real_strain)) and not np.any(np.isnan(imag_strain)) and not np.any(np.isnan(psd_values)), "Found NaNs"
            
        print(f"Saving {ifo.name} data to npz file")
        strain = real_strain + 1j * imag_strain
        np.savez(os.path.join(outdir, f"{ifo.name}_datadump.npz"), frequencies=frequencies, data=strain, psd=psd_values)
        
    return 1
        
def main():
    
    failed_events = []
    
    for event_id in event_ids:
        print(f"\n\n\n--- Processing for event {event_id} ---\n\n\n")
        metadata = load_event_metadata(event_id)
        if metadata == {}:
            print(f"Metadata failed, skipping")
            continue
        
            failed_events.append(event_id)
        
        else:
            # Save the metadata
            outdir = f"./outdir/{event_id}/"
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            with open(os.path.join(outdir, "metadata.json"), "w") as f:
                json.dump(metadata, f)
        
            success = get_data_and_psd(metadata["outdir"], outdir = outdir)
            if not success:
                failed_events.append(event_id)
                
    print(f"There were {len(failed_events)}/{len(event_ids)} failed events: {failed_events}")
        
if __name__ == "__main__":
    main()