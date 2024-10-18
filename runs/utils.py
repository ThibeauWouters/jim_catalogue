import argparse

def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Perform an injection recovery.",
        add_help=add_help,
    )
    
    ### Required arguments to run the PE
    parser.add_argument(
        "--outdir",
        type=str,
        default="./outdir/",
        help="Output directory for the injection.",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        help="ID of the event on which we run PE",
    )
    
    ### Hyperparameters come here
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=20,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--n-loop-training",
        type=int,
        default=100,
        help="Number of loops for training",
    )
    parser.add_argument(
        "--n-loop-production",
        type=int,
        default=20,
        help="Number of loops for production",
    )
    parser.add_argument(
        "--n-local-steps",
        type=int,
        default=10,
        help="Number of local steps",
    )
    parser.add_argument(
        "--n-global-steps",
        type=int,
        default=1000,
        help="Number of global steps",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=500,
        help="Number of chains",
    )
    parser.add_argument(
        "--n-max-examples",
        type=int,
        default=30000,
        help="Number of maximum examples",
    )
    parser.add_argument(
        "--n-flow-sample",
        type=int,
        default=100000,
        help="Number of flow samples",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30000,
        help="Batch size",
    )
    parser.add_argument(
        "--use-global",
        type=bool,
        default=True,
        help="Use global",
    )
    parser.add_argument(
        "--keep-quantile",
        type=float,
        default=0.0,
        help="Keep quantile",
    )
    parser.add_argument(
        "--train-thinning",
        type=int,
        default=1,
        help="Train thinning",
    )
    parser.add_argument(
        "--output-thinning",
        type=int,
        default=10,
        help="Output thinning",
    )
    
    return parser