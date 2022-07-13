from pathlib import Path

from gym_PBN.envs import PBNTargetEnv
from gym_PBN.envs.bittner.utils import spawn
from gym_PBN.utils.eval import compute_ssd_hist

if __name__ == "__main__":
    # Step 1 - Inference
    # fmt: off
    include_ids = [234237, 324901, 759948, 25485, 266361, 108208, 130057]
    # fmt: on

    predictor_sets_path = (
        Path(__file__).parent / "gym_PBN" / "envs" / "bittner" / "data"
    )
    genedata = predictor_sets_path / "genedata.xls"

    graph = spawn(
        file=genedata,
        total_genes=200,
        include_ids=include_ids,
        bin_method="kmeans",
        n_predictors=5,
        predictor_sets_path=predictor_sets_path,
    )

    goal_config = {
        "target_nodes": [324901],
        "intervene_on": [234237],
        "target_node_values": ((0,),),
        "undesired_node_values": tuple(),
        "horizon": 11,
    }

    env = PBNTargetEnv(graph, goal_config, "Bittner-200")

    # Step 2 - Evaluation
    ssd = compute_ssd_hist(env, resets=300, iters=1_200_000)
    print(ssd)
