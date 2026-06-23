import numpy

DEMENSIONS_PER_STEP = 4

# bwoを実行するメソッド
def run_bwo(vco_config):
    print("hello bwo")
    current_results = []
    current = vco_config.target_current

    population = []
    for _ in range(vco_configs.population_size):
        vc_individual = init_whale(vco_config)

        deap_individual = creator.Individual([
            vc_individual
        ])

        population.append(deap_individual)


# 探索個体（シロイルカ）の初期化を行うメソッド
def init_whale(config):
    # シロイルカの初期一の作成
    position = np.random.uniform(
        0.0,    # 最小値
        1.0,    # 最大値
        # プロトコルの個数 x 次元数(type,start,end,duration)
        config.steps_in_protocol * DEMENSIONS_PER_STEP
    )

    # 初期状態のtypeをStep/Rampに分ける
    for i in range(config.steps_in_protocol):
        type_index = i * DEMENSIONS_PER_STEP
        position[type_index] = random.choice([0.0, 1.0])

    return position

# BWOの処理を行うための初期化を行い、run_bwoを呼び出すメソッド
def start_bwo(vco_config):
    print("start bwo")