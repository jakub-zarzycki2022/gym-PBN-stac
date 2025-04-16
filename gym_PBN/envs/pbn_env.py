import pickle

from gym_PBN.envs.bittner.pbn_graph import PBNNode, PBNGraph
from gym_PBN.envs.pbn_target_multi import PBNTargetMultiEnv


class PBNEnv(PBNTargetMultiEnv):
    NAME = "PBN"

    def __init__(
            self,
            N=72,
            render_mode: str = "human",
            render_no_cache: bool = False,
            name: str = NAME,
            horizon: int = 100,
            end_episode_on_success: bool = True,
            logic_functions=None,
            genes=None,
            min_attractors=3,
    ):
        self.N = N
        print(f"its me, PBN-{self.N}")
        self.path = f"attractors/{self.N}_{1}_attractors.pkl"
        if not name:
            self.NAME = f"{self.NAME}-{N}"
            name = self.NAME

        graph = PBNGraph(genes, logic_functions)

        super().__init__(
            graph,
            {},
            render_mode,
            render_no_cache,
            name,
            end_episode_on_success,
            horizon=20,
            min_attractors=min_attractors,
        )

        self.divided_attractors = []
        self.forbidden_actions = []

        input_nodes = []
        # self.initial_values = []

        # self.initial_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #
        # for node_name in ["IL27RA", "IL27_e", "GP130", "Galpha_QL", "IL2RB", "CGC", "Galpha_iL", "MHC_II", "APC",
        #                   "IL18_e", "IL9_e", "IFNB_e", "ECM", "IL21_e", "alpha_13L", "IL10RA", "IL10RB", "IL10_e",
        #                   "IL15_e", "B7", "IFNGR1", "IFNGR2", "IFNG_e", "CAV1_ACTIVATOR", "GalphaS_L", "IL4_e",
        #                   "IL6_e", "IL6RA", "TGFB_e", "IL22_e", "IL2_e", "IL23_e", "IL15RA", "IL12_e"]:

        # bortezomib:
        # input_nodes = ["XX", "SHP1", "TNFAR", "Bort", "TNFA"]
        # self.initial_values = [0, 0, 0, 0, 0]

        # bortezomib_general:
        input_nodes = []
        self.initial_values = []

        # bladder
            # input_nodes = ["v_GrowthInhibitors", "v_Growth_Arrest", "v_Proliferation", "v_DNAdamage"]
        # self.initial_values = [0, 1, 1, 1]

        # bladder2
        # input_nodes = ["v_EGFR_stimulus", "v_GrowthInhibitors"]
        # self.initial_values = [0, 1]

        # mapk
        # input_nodes = ["v_EGFR_stimulus", "v_FGFR3_stimulus"]
        # self.initial_values = [0, 0]

        # mcf7
        # input_nodes = ['v_ABL2', 'v_DLL_i', 'v_EGF', 'v_ES', 'v_IGF1', 'v_IL6R', 'v_INS', 'v_NRG1', 'v_PG', 'v_WNT1']
        # self.inital_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # input_nodes = ["XX", "SHP1", "TNFAR", "Bort", "TNFA"]

        #CD4+
        # input_nodes = ["IL27RA", "IL27_e", "GP130", "Galpha_QL", "IL2RB", "CGC", "Galpha_iL", "MHC_II", "APC",
        #                   "IL18_e", "IL9_e", "IFNB_e", "ECM", "IL21_e", "alpha_13L", "IL10RA", "IL10RB", "IL10_e",
        #                   "IL15_e", "B7", "IFNGR1", "IFNGR2", "IFNG_e", "CAV1_ACTIVATOR", "GalphaS_L", "IL4_e",
        #                   "IL6_e", "IL6RA", "TGFB_e", "IL22_e", "IL2_e", "IL23_e", "IL15RA", "IL12_e"]
        # self.initial_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # assert (len(input_nodes) == len(self.initial_values))


        #tdiff_jun
        # input_nodes = ["APC", "IFNB_e", "IFNG_e", "IL2_e", "IL4_e", "IL6_e", "IL10_e", "IL12_e", "IL15_e", "IL21_e",
        #                "IL23_e", "IL27_e", "TGFB_e", "IFNGR1", "IFNGR2", "GP130", "IL6RA", "CGC", "IL12RB2", "IL10RB",
        #                "IL10RA", "IL15RA", "IL2RB", "IL27RA"]
        #
        # self.initial_values = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        #random200
        # inpurt_nodes = []
        # self.initial_values = []

        for node_name in input_nodes:
            for i in range(len(self.graph.nodes)):
                if self.graph.nodes[i].name == f"{node_name}" or self.graph.nodes[i].name == f"v_{node_name}":
                    self.forbidden_actions.append(i)

        # print(sorted([node.name for node in self.graph.nodes]))

        out_node_names = []
        # bortezomib
        self.out_node_values = [0, 1, 1]
        out_node_names = ["JNK", "p21", "Cas3"]

        # aba
        # out_node_names = ['v_Closure']

        # to już nie bortezomib
        # out_node_names = ["Dec2", "SHP2", "GATA3"]

        #cd4+
        # out_node_names = ["v_GATA3"]
        # self.out_node_values = [1]

        #tdiff_jun
        # out_node_names = ["TBET", "GATA3", "RORGT", "FOXP3"]
        # self.out_node_values = [1, 0, 1, 1]

        #mcf7
        # out_node_names = ["v_CASP3", "v_E2F1"]


        #tlgl
        # out_node_names = ["v_Apoptosis"]
        # self.out_node_values = [1]

        #MAPK
        # out_node_names = ["v_Apoptosis"]
        # self.out_node_values = [1]

        #random 200
        # out_node_names = ["x3"]
        # self.out_node_values = [1]

        #bladder
        # out_node_names = ["v_Apoptosis_b1"]
        # self.out_node_values = [1]

        self.out_nodes = []

        # for node in self.graph.nodes:
        #     print(node.name)
        # # raise ValueError
        # print(self.out_nodes)

        for node_name in out_node_names:
            for i in range(len(self.graph.nodes)):
                if self.graph.nodes[i].name == f"{node_name}":
                    self.out_nodes.append(i)

        print("init nodes are: ", self.forbidden_actions[:34])
        print("out nodes are: ", self.out_nodes)
        # for i in self.out_nodes:
        #     print(self.graph.nodes[i].name)

        try:
            print(f"try to load: \n{self.path}")
            with open(self.path, "rb") as f:
                attractors = pickle.load(f)
                self.all_attractors = attractors
                self.attractor_set = {a[0] for a in self.all_attractors}
                self.divided_attractors = [a[0] for a in self.all_attractors if
                                           not (self.in_target(a[0]))]

                self.target_attractors = [a[0] for a in self.all_attractors if
                                          self.in_target(a[0])]
        except FileNotFoundError:
            print('calculating new attractors')
            self.target_attractors = []

            self.all_attractors += [[s] for s in self.statistical_attractors(3)]
            self.divided_attractors = [a[0] for a in self.all_attractors if not (self.in_target(a[0]))]

            self.target_attractors = [a[0] for a in self.all_attractors if self.in_target(a[0])]

            with open(self.path, "wb+") as f:
                pickle.dump(self.all_attractors, f)

        while len(self.target_attractors) == 0:
            for at in self.all_attractors:
                print([at[0][i] for i in range(65, 100)])

            for i, node in enumerate(self.graph.nodes):
                print(i, node.name)
            for at in self.all_attractors:
                print([at[0][x] for x in self.out_nodes])
            raise ValueError
            # print(f"got {len(self.all_attractors)} attractors so far but no target")

            self.reset()
            state = list([1] * 67)

            for i in range(len(self.initial_values)):
                # print('i', i)
                # print('fa', self.forbidden_actions[i])
                # print('s', len(state))
                state[self.forbidden_actions[i]] = self.initial_values[i]

            for i, v in enumerate(self.out_nodes):
                state[v] = self.out_node_values[i]

            # print('state of il12e is :', state[self.forbidden_actions[33]])
            # print(state)
            #
            # print('set manually')
            self.graph.setState(state)
            # print('call step')
            self.step([])
            # self.step([])
            # print('step called')

            astate = tuple(self.get_state())
            # print('got: ', [astate[i] for i in self.out_nodes])
            # print('exp: ', self.out_node_values)

            if self.in_target(astate):
                self.target_attractors.append(astate)
            else:
                pass
                # print('wtf? still no attractor')


        # for attractor in self.all_attractors:
        #     print([attractor[0][self.forbidden_actions[i]] for i in range(5)], [attractor[0][out_node] for out_node in self.out_nodes])

        print(f'got {len(self.all_attractors)} attractors')
        self.attractor_set = {a[0] for a in self.all_attractors}
        print(f'got {len(self.attractor_set)} after dedup attractors')

        print(len(self.divided_attractors), len(self.target_attractors))
        
        for attractor in self.all_attractors:
            print(attractor[0][10], attractor[0][14], self.is_singleton(attractor[0]))

        # print(self.initial_values)
        # raise ValueError
            

        self.attracting_states.update([s[0] for s in self.all_attractors])

        self.attractor_count = len(self.all_attractors)
        self.probabilities = [1 / self.attractor_count] * self.attractor_count

        # print(self.all_attractors)
        self.forbidden_actions += self.out_nodes

    def is_attracting_state(self, state):
        state = tuple(state)

        return state in self.attractor_set

    def in_target(self, observation):
        for i in range(len(self.out_node_values)):
            if observation[self.out_nodes[i]] != self.out_node_values[i]:
                return False
        return True

        # if observation[nodes[0]] + observation[nodes[1]] == 0:
        #     return False

    def is_singleton(self, state):
        state = tuple(state)
        old_state = state

        self.graph.setState(old_state)

        for node in self.graph.nodes:
            self.graph.setState(old_state)
            node.step(old_state)
            x = (self.graph.getState() == old_state)
            if not x:
                print('not a singleton')

                if not self.is_attracting_state(self.graph.getState()):
                    for _ in range(0000):
                        self.graph.step()
                        if self.graph.getState() == old_state:
                            return True

                    print('was ', old_state)
                    print('got ', self.graph.getState())
                    for i in range(53):
                        if self.graph.getState()[i] != old_state[i]:
                            print('node ', i, "name ", self.graph.nodes[i].name, self.graph.nodes[i].value)
                            for p in self.graph.nodes[i].predictors[0]:
                                print(self.graph.nodes[p].name, self.graph.nodes[p].value)
                    return False

        return True

