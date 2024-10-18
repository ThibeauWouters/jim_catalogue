import os 
# # --- For running on CIT
# import psutil
# p = psutil.Process()
# p.cpu_affinity([0])
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.10"
# # ---

import time
import utils
# import requests
# import h5py
import json
import numpy as np

print("Importing JAX")
import jax
import jax.numpy as jnp
print("Importing JAX successful")

print(f"Checking for CUDA: JAX devices {jax.devices()}")

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
from jimgw.single_event.detector import H1, L1, V1, GroundBased2G
from jimgw.single_event.likelihood import TransientLikelihoodFD, HeterodynedTransientLikelihoodFD
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

GWTC_EVENT_CSV = "../data/GWTC3_confident.csv"

def run_pe(args: argparse.Namespace,
           verbose: bool = True,
           delta_Mc: float = 10.0):
    
    total_time_start = time.time()
    print("args")
    print(args)
    
    # Make the outdir
    if not os.path.exists(os.path.join(args.outdir, args.event_id)):
        os.makedirs(os.path.join(args.outdir, args.event_id))
        
    # Fetch our event information from the CSV
    df = pd.read_csv(GWTC_EVENT_CSV)
    event = df[df["commonName"] == args.event_id].iloc[0]
    Mc = event["chirp_mass"]
    
    # Get the preprocessed data
    data_dir = f"../data/outdir/{args.event_id}/"
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    if verbose:
        print("metadata")
        print(metadata)
    
    duration = float(metadata["duration"])
    post_trigger = 2
    gps = metadata["trigger_time"]
    trigger_time = gps
    fmin: dict[str, float] = metadata["f_min"]
    fmax: dict[str, float] = metadata["f_max"]
    
    fmin = np.min(list(fmin.values()))
    fmax = np.min(list(fmax.values()))
    
    fmin = max(fmin, 20.0)
    fmax = min(fmax, 2048.0)
    
    ifos_list_string = metadata["detectors"]
    
    if verbose:
        print("fmin")
        print(fmin)
        
        print("fmax")
        print(fmax)
    
    # Load the HDF5 files from the ifos dict url and open it:
    ifos: list[GroundBased2G] = []
    for i, ifo_string in enumerate(ifos_list_string):
        
        print("Adding interferometer ", ifo_string)
        eval(f'ifos.append({ifo_string})')
        
        # Load the data
        data_file = os.path.join(data_dir, f"{ifo_string}_datadump.npz")
        print(f"Loading data for {ifo_string} from {data_file}")
        datadump = np.load(data_file)
    
        frequencies, data, psd = datadump["frequencies"], datadump["data"], datadump["psd"]
        
        mask = (frequencies >= fmin) & (frequencies <= fmax)
        frequencies = frequencies[mask]
        data = data[mask]
        psd = psd[mask]
    
        ifos[i].frequencies = frequencies
        ifos[i].data = data
        ifos[i].psd = psd
        
        if verbose:
            print(f"Checking data for {ifo_string}")
            print(f"Data shape: {ifos[i].data.shape}")
            print(f"PSD shape: {ifos[i].psd.shape}")
            print(f"Frequencies shape: {ifos[i].frequencies.shape}")
            
            print(f"Data: {ifos[i].data}")
            print(f"PSD: {ifos[i].psd}")
            print(f"Frequencies: {ifos[i].frequencies}")
    
    if verbose:
        print(f"Running PE on event {args.event_id}")
        print(f"Duration: {duration}")
        print(f"GPS: {gps}")
        print(f"Chirp mass: {Mc}")
    
    waveform = RippleIMRPhenomPv2(f_ref=20)

    ###########################################
    ########## Set up priors ##################
    ###########################################

    prior = []

    # Mass prior
    # TODO: use the chirp mass from the PE summary here
    M_c_min, M_c_max = Mc - delta_Mc, Mc + delta_Mc
    q_min, q_max = 0.125, 1.0
    Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
    q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

    prior = prior + [Mc_prior, q_prior]

    # Spin prior
    s1_prior = UniformSpherePrior(parameter_names=["s1"])
    s2_prior = UniformSpherePrior(parameter_names=["s2"])
    iota_prior = SinePrior(parameter_names=["iota"])

    prior = prior + [
        s1_prior,
        s2_prior,
        iota_prior,
    ]

    # Extrinsic prior
    dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
    t_c_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
    phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
    psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
    ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
    dec_prior = CosinePrior(parameter_names=["dec"])

    prior = prior + [
        dL_prior,
        t_c_prior,
        phase_c_prior,
        psi_prior,
        ra_prior,
        dec_prior,
    ]

    prior = CombinePrior(prior)

    # Defining Transforms

    sample_transforms = [
        DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
        GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
        GeocentricArrivalTimeToDetectorArrivalTimeTransform(tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]),
        SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
        BoundToUnbound(name_mapping = (["M_c"], ["M_c_unbounded"]), original_lower_bound=M_c_min, original_upper_bound=M_c_max),
        BoundToUnbound(name_mapping = (["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max),
        BoundToUnbound(name_mapping = (["s1_phi"], ["s1_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
        BoundToUnbound(name_mapping = (["s2_phi"], ["s2_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
        BoundToUnbound(name_mapping = (["iota"], ["iota_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
        BoundToUnbound(name_mapping = (["s1_theta"], ["s1_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
        BoundToUnbound(name_mapping = (["s2_theta"], ["s2_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
        BoundToUnbound(name_mapping = (["s1_mag"], ["s1_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
        BoundToUnbound(name_mapping = (["s2_mag"], ["s2_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
        BoundToUnbound(name_mapping = (["phase_det"], ["phase_det_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
        BoundToUnbound(name_mapping = (["psi"], ["psi_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        BoundToUnbound(name_mapping = (["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        BoundToUnbound(name_mapping = (["azimuth"], ["azimuth_unbounded"]), original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    ]

    likelihood_transforms = [
        MassRatioToSymmetricMassRatioTransform,
        SphereSpinToCartesianSpinTransform("s1"),
        SphereSpinToCartesianSpinTransform("s2"),
    ]

    # TODO: memory issues
    # likelihood = TransientLikelihoodFD(
    #     ifos, waveform=waveform, trigger_time=gps, duration=duration, post_trigger_duration=post_trigger
    # )
    
    likelihood = HeterodynedTransientLikelihoodFD(ifos, 
                                                  waveform=waveform, 
                                                  n_bins = 1000, 
                                                  trigger_time=trigger_time, 
                                                  duration=duration, 
                                                  post_trigger_duration=post_trigger, 
                                                  prior=prior, 
                                                  sample_transforms=sample_transforms,
                                                  likelihood_transforms=likelihood_transforms,
                                                  popsize=10,
                                                  n_steps=50)


    mass_matrix = jnp.eye(prior.n_dim)
    # mass_matrix = mass_matrix.at[1, 1].set(1e-3)
    # mass_matrix = mass_matrix.at[9, 9].set(1e-3)
    local_sampler_arg = {"step_size": mass_matrix * 1e-3}

    Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

    import optax

    n_epochs = 20
    n_loop_training = 100
    total_epochs = n_epochs * n_loop_training
    start = total_epochs // 10
    learning_rate = optax.polynomial_schedule(
        1e-3, 1e-4, 4.0, total_epochs - start, transition_begin=start
    )

    jim = Jim(
        likelihood,
        prior,
        sample_transforms=sample_transforms,
        likelihood_transforms=likelihood_transforms,
        n_loop_training=n_loop_training,
        n_loop_production=args.n_loop_production,
        n_local_steps=args.n_local_steps,
        n_global_steps=args.n_global_steps,
        n_chains=args.n_chains,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        n_max_examples=args.n_max_examples,
        n_flow_sample=args.n_flow_sample,
        momentum=args.momentum,
        batch_size=args.batch_size,
        use_global=True,
        keep_quantile=args.keep_quantile,
        train_thinning=args.train_thinning,
        output_thinning=args.output_thinning,
        local_sampler_arg=local_sampler_arg,
        # strategies=[Adam_optimizer,"default"],
    )

    jim.sample(jax.random.PRNGKey(42))
    
    # Postprocessing comes here
    samples: dict = jim.get_samples()
    jnp.savez(os.path.join(args["outdir"], args["event_id"], "samples.npz"), **samples)

    total_time_end = time.time()
    print(f"Time taken: {total_time_end - total_time_start} seconds = {(total_time_end - total_time_start) / 60} minutes")

def main():
    parser = utils.get_parser()
    args = parser.parse_args()
    run_pe(args)
    
if __name__ == "__main__":
    main()