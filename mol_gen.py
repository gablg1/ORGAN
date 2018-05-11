import organ
from organ import ORGAN

organ_params = {
    'PRETRAIN_GEN_EPOCHS': 250, 'PRETRAIN_DIS_EPOCHS': 10, 'MAX_LENGTH': 60, 'LAMBDA': 0.5, "DIS_EPOCHS": 2, 'SAMPLE_NUM': 6400, 'WGAN':True}

# hyper-optimized parameters
disc_params = {"DIS_L2REG": 0.2, "DIS_EMB_DIM": 32, "DIS_FILTER_SIZES": [
    1, 2, 3, 4, 5, 8, 10, 15], "DIS_NUM_FILTERS": [50, 50, 50, 50, 50, 50, 50, 75], "DIS_DROPOUT": 0.75}

organ_params.update(disc_params)

model = ORGAN('qm9-5k', 'mol_metrics', params=organ_params)
model.load_training_set('data/qm9_5k.csv')
# model.load_prev_pretraining('pretrain_ckpt/qm9-5k_pretrain_ckpt')
model.set_training_program(
    ['druglikeliness'], [100])
model.load_metrics()
# model.load_prev_training(ckpt='qm9-5k_20.ckpt')
model.train()
