from pathlib import Path

from gym_PBN.envs import PBNTargetEnv
from gym_PBN.envs.bittner.utils import spawn
from gym_PBN.utils.eval import compute_ssd_hist

# Step 1 - Inference
# fmt: off
include_ids = [234237, 324901, 759948, 25485, 324700, 43129, 266361, 108208, 40764, 130057, 39781, 49665, 39159, 23185,417218, 31251, 343072, 142076, 128100, 376725, 112500, 241530, 44563, 36950, 812276, 51018, 306013, 418105]
# fmt: on

predictor_sets_path = Path(__file__).parent / "gym_PBN" / "envs" / "bittner" / "data"
genedata = predictor_sets_path / "genedata.xls"

graph = spawn(
    file=genedata,
    total_genes=28,
    include_ids=include_ids,
    bin_method="median",
    n_predictors=15,
    predictor_sets_path=predictor_sets_path,
)

goal_config = {
    "target_nodes": [324901],
    "intervene_on": [234237],
    "target_node_values": ((0,),),
    "undesired_node_values": tuple(),
    "horizon": 11,
}

env = PBNTargetEnv(graph, goal_config, "Bittner-28")

# Step 2 - Evaluation
ssd = compute_ssd_hist(env, resets=300, iters=100_000)
print(ssd)
