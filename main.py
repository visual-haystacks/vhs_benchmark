import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import create_directory, run_eval, run_detailed_eval


# Set up logging + Disable annoying HTTPX logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.CRITICAL)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.info("Starting the evaluation process...")

    # Resolve all interpolations
    OmegaConf.resolve(cfg)
    # Convert the fully resolved configuration to YAML
    yaml_cfg = OmegaConf.to_yaml(cfg)
    logging.info(f"Using configuration:\n{yaml_cfg}")
    create_directory(cfg.basic.output_dir)

    solver = hydra.utils.instantiate(cfg.solver)
    logging.info("Solver instantiated successfully.")

    for img in cfg.basic.image_counts:
        try:
            test_file = os.path.join(
                cfg.basic.test_file_base, f"visual_haystack_{img}.json"
            )
            sub_dir_name = os.path.join(cfg.basic.output_dir, f"{img}_images")
            os.makedirs(sub_dir_name, exist_ok=True)
            logging.info(
                f"Processing case with {img} images: data written to {sub_dir_name}"
            )
            if cfg.basic.mode == "multi_needle":
                solver.run_fast(test_file, sub_dir_name)
                logging.info(
                    f"Running evaluation for ({cfg.solver.name},{cfg.basic.mode})"
                )
                run_eval(sub_dir_name, cfg.basic.mode)
            else:
                solver.run_detailed(test_file, sub_dir_name)
                logging.info(
                    f"Running evaluation for ({cfg.solver.name},{cfg.basic.mode})"
                )
                run_detailed_eval(sub_dir_name, cfg.basic.mode)

        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
