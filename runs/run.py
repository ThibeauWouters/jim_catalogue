# --- For running on CIT
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
# ---

import time
import utils
import requests
import h5py

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.jim import Jim
from jimgw.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)
from jimgw.single_event.utils import Mc_q_to_m1_m2
from flowMC.strategy.optimization import optimization_Adam

jax.config.update("jax_enable_x64", True)

import argparse
import pandas as pd

GWTC_EVENT_CSV = "../metadata/GWTC3_confident.csv"

def run_pe(args: argparse.Namespace,
           verbose: bool = False):
    
    # Make dict from NameSpace
    print("args")
    print(args)
    
    if not os.path.exists(os.path.join(args.outdir, args.event_id)):
        os.makedirs(os.path.join(args.outdir, args.event_id))
    
    # Fetch event and metadata
    df = pd.read_csv(GWTC_EVENT_CSV)
    event = df[df["commonName"] == args.event_id].iloc[0]
    
    url = event["jsonurl"]
    print(f"Loading JSON url from {url}")
    r = requests.get(url)
    json_data = r.json()
    json_data = json_data["events"][args.event_id + "-v1"]
    
    # TODO: get rid of this?
    # # Grab chirptime and set the duration to be nearest rounded to above power of two of chirp time
    # chirp_time = event["chirp_time"]
    # duration = 2 ** jnp.ceil(jnp.log2(chirp_time))
    # duration = max(duration, 4.0)
    
    ### Handle PSD data and posterior samples from GWOSC PE results
    all_pe: dict[str, dict] = json_data["parameters"]
    for _, pe in all_pe.items():
        if pe["is_preferred"] == True:
            break
    
    print("PE result metadata")
    print(pe)
    
    # Load the HDF5 file from the pe url and open it:
    pe_url = pe["data_url"]
    response = requests.get(pe_url, stream=True)
    
    # Get the desired filename
    local_filename = pe_url.split("/")[-2]
    local_filename = os.path.join(args.outdir, args.event_id, local_filename)
        
    # Check if the request was successful
    response = requests.get(pe_url, stream=True)
    if response.status_code == 200:
        # Open the file in write-binary mode and write the content
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        print(f"File saved as {local_filename}")
    else:
        raise ValueError(f"Failed to download file. Status code: {response.status_code}")
    
    # Open the file to check the contents
    with h5py.File(local_filename, "r") as f:
        print("Keys: %s" % f.keys())
        
        data = f["C01:Mixed"]
        psds = data["psds"]
        print(psds.keys())
    
    ### Handle strain data
    strain: list[dict] = json_data["strain"]
    new_strain = []
    for d in strain:
        if d["sampling_rate"] != 16384 and d["format"] == "hdf5" and d["duration"] < 4096:
            new_strain.append(d)
    strain = new_strain
    
    if verbose:
        print("strain")
        print(strain)
    
    if len(strain) == 0:
        raise ValueError("No strain data found for this event, something went wrong?")
    
    # Get ifos of this event from strain
    ifos_dict: dict[str, str] = {}
    for d in strain:
        ifos_dict[d["detector"]] = d["url"]
        
    if verbose:
        print("ifos_dict")
        print(ifos_dict)
    
    # Load the HDF5 files from the ifos dict url and open it:
    ifos = []
    for ifo, url in ifos_dict.items():
        
        # Get the desired filename
        local_filename = url.split("/")[-1]
        local_filename = os.path.join(args.outdir, args.event_id, local_filename)
        
        # Load it with requests
        print(f"Loading {ifo} from {url}, saving it to {local_filename}")
        response = requests.get(url, stream=True)

        # Open the local file in write-binary mode
        with open(local_filename, 'wb') as file:
            # Write the content in chunks to avoid loading large files into memory
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)        

        with h5py.File(local_filename, "r") as f:
            strain_data = f["strain"]["Strain"]
            
            if verbose:
                print("strain_data")
                print(strain_data)
            
            
        # print("Adding interferometer ", ifo_string)
        # eval(f'ifos.append({ifo_string})')
        
    
        # # TODO: Set the data
        # ifos[i].frequencies = None
        # ifos[i].data = None
        # ifos[i].psd = None
            
        ifos.append(None)
    
    gps = event["GPS"]
    Mc = event["chirp_mass"]
    
    if verbose:
        print(f"Running PE on event {args.event_id}")
        # print(f"Chirp time: {chirp_time}")
        # print(f"Duration: {duration}")
        print(f"GPS: {gps}")
        print(f"Chirp mass: {Mc}")
    
    ##################################
    ########## Grab data #############
    ##################################

    # total_time_start = time.time()
    
    # # first, fetch a 4s segment centered on GW150914
    # gps = gps
    # post_trigger = 2
    
    # # TODO: fetch from the PE summary as well
    # fmin = 20.0
    # fmax = 1024.0

    # # Setup interferometers
    
    # H1.load_data(gps, duration - post_trigger, post_trigger, fmin, fmax, psd_pad=16, tukey_alpha=0.2)
    # L1.load_data(gps, duration - post_trigger, post_trigger, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

    # waveform = RippleIMRPhenomPv2(f_ref=20)

    # ###########################################
    # ########## Set up priors ##################
    # ###########################################

    # prior = []

    # # Mass prior
    # # TODO: use the chirp mass from the PE summary here
    # M_c_min, M_c_max = Mc - 10.0, Mc + 10.0
    # q_min, q_max = 0.125, 1.0
    # Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
    # q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

    # prior = prior + [Mc_prior, q_prior]

    # # Spin prior
    # s1_prior = UniformSpherePrior(parameter_names=["s1"])
    # s2_prior = UniformSpherePrior(parameter_names=["s2"])
    # iota_prior = SinePrior(parameter_names=["iota"])

    # prior = prior + [
    #     s1_prior,
    #     s2_prior,
    #     iota_prior,
    # ]

    # # Extrinsic prior
    # dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
    # t_c_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
    # phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
    # psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
    # ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
    # dec_prior = CosinePrior(parameter_names=["dec"])

    # prior = prior + [
    #     dL_prior,
    #     t_c_prior,
    #     phase_c_prior,
    #     psi_prior,
    #     ra_prior,
    #     dec_prior,
    # ]

    # prior = CombinePrior(prior)

    # # Defining Transforms

    # sample_transforms = [
    #     DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
    #     GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    #     GeocentricArrivalTimeToDetectorArrivalTimeTransform(tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]),
    #     SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    #     BoundToUnbound(name_mapping = (["M_c"], ["M_c_unbounded"]), original_lower_bound=M_c_min, original_upper_bound=M_c_max),
    #     BoundToUnbound(name_mapping = (["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max),
    #     BoundToUnbound(name_mapping = (["s1_phi"], ["s1_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    #     BoundToUnbound(name_mapping = (["s2_phi"], ["s2_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    #     BoundToUnbound(name_mapping = (["iota"], ["iota_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    #     BoundToUnbound(name_mapping = (["s1_theta"], ["s1_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    #     BoundToUnbound(name_mapping = (["s2_theta"], ["s2_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    #     BoundToUnbound(name_mapping = (["s1_mag"], ["s1_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
    #     BoundToUnbound(name_mapping = (["s2_mag"], ["s2_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
    #     BoundToUnbound(name_mapping = (["phase_det"], ["phase_det_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    #     BoundToUnbound(name_mapping = (["psi"], ["psi_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
    #     BoundToUnbound(name_mapping = (["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
    #     BoundToUnbound(name_mapping = (["azimuth"], ["azimuth_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    # ]

    # likelihood_transforms = [
    #     MassRatioToSymmetricMassRatioTransform,
    #     SphereSpinToCartesianSpinTransform("s1"),
    #     SphereSpinToCartesianSpinTransform("s2"),
    # ]


    # likelihood = TransientLikelihoodFD(
    #     [H1, L1], waveform=waveform, trigger_time=gps, duration=duration, post_trigger_duration=post_trigger
    # )


    # mass_matrix = jnp.eye(prior.n_dim)
    # # mass_matrix = mass_matrix.at[1, 1].set(1e-3)
    # # mass_matrix = mass_matrix.at[9, 9].set(1e-3)
    # local_sampler_arg = {"step_size": mass_matrix * 1e-3}

    # Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

    # import optax

    # n_epochs = 20
    # n_loop_training = 100
    # total_epochs = n_epochs * n_loop_training
    # start = total_epochs // 10
    # learning_rate = optax.polynomial_schedule(
    #     1e-3, 1e-4, 4.0, total_epochs - start, transition_begin=start
    # )

    # jim = Jim(
    #     likelihood,
    #     prior,
    #     sample_transforms=sample_transforms,
    #     likelihood_transforms=likelihood_transforms,
    #     n_loop_training=n_loop_training,
    #     n_loop_production=args["n_loop_production"],
    #     n_local_steps=args["n_local_steps"],
    #     n_global_steps=args["n_global_steps"],
    #     n_chains=args["n_chains"],
    #     n_epochs=n_epochs,
    #     learning_rate=learning_rate,
    #     n_max_examples=args["n_max_examples"],
    #     n_flow_sample=args["n_flow_sample"],
    #     momentum=args["momentum"],
    #     batch_size=args["batch_size"],
    #     use_global=True,
    #     keep_quantile=args["keep_quantile"],
    #     train_thinning=args["train_thinning"],
    #     output_thinning=args["output_thinning"],
    #     local_sampler_arg=local_sampler_arg,
    #     # strategies=[Adam_optimizer,"default"],
    # )

    # jim.sample(jax.random.PRNGKey(42))
    
    # # Postprocessing comes here
    # samples: dict = jim.get_samples()
    # jnp.savez(os.path.join(args["outdir"], args["event_id"], "samples.npz"), **samples)


def main():
    parser = utils.get_parser()
    args = parser.parse_args()
    run_pe(args)
    
if __name__ == "__main__":
    main()