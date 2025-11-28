import argparse
from pathlib import Path
from cub.dataset import load_data
import torch

def get_unique_output_dir(base_dir: Path) -> Path:
    """ Return a uniqe dir if already exists.
    
    Arguments:
        base_dir (Path): The base directory to check.
    """
    if not base_dir.exists():
        return base_dir
    
    counter = 1
    while True:
        new_dir = Path(f"{base_dir}_{counter}")
        if not new_dir.exists():
            return new_dir
        if counter >= 20:
            raise RuntimeError("Could not find a unique directory name after 20 attempts.")
        
        counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # store true if flag is set
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test argument"
    )

    parser.add_argument(
        "--batch_size", 
        type=int,
        default=8,
        help="Batch size for data loading"
    )

    parser.add_argument(
        "--model_path", 
        type=str,
        required=True,
        help="Path to the PyTorch model file"
    )

    # Todo: auto choices from available models
    parser.add_argument(
        "--model_type", 
        type=str,
        required=True,
        help="Type of the model architecture"
    )

    args = parser.parse_args()

    # define additional args
    args.data_dir = Path("data/CUB_processed/class_attr_data_10")
    args.batch_size = 8
    args.image_dir = Path("data/CUB_200_2011/images")
    args.mode = "XCY"

    # load the data
    val_loader = load_data(args, "val")

    # determine the part which should be used for saliency


    # iterate through the data
    for batch in val_loader:
        images, labels, attr = batch
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Attributes shape: {len(attr)}")

        print(attr)

        if args.model_type == "CBM":
            from saliency.wrapper import WrapperCUB as ModelWrapper
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # load the model
        loaded_model = torch.load(
            args.model_path, 
            map_location="cpu", 
            weights_only=False
        )

        model_wrapper = ModelWrapper(args.model_path)
        




        break  


