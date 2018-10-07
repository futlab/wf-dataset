import json
from os import listdir
from os.path import join


def load_models(folder):
    model_states = {}
    for f in listdir(folder):
        if f.endswith('_state.json'):
            try:
                s = json.load(open(join(folder, f)))
                model_states.update({f[:-11] : s})
            except json.JSONDecodeError:
                print('Unable to read model state: ' + f)
    return model_states


def get_best_models(ms, trim=10):
    def bl(m):
        return ms[m]['best_loss']
    bm = sorted(list(ms.keys()), key=bl)
    if trim is not None and len(bm) > trim:
        bm = bm[:trim]
    return bm


models = load_models('models-sam')
best = get_best_models(models, None)
for i in range(0, len(best) - 1):
    model_name = best[i]
    model = models[model_name]
    print('%3d: (vl %.4f) %s <- %s (epochs: %d, params: %d)' % (i + 1, model.get('best_loss', None), model_name, model.get('parent', '???'), model.get('epoch', '0'), model.get('param_count', 0)))
