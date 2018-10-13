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


def get_best_models(ms, trim=None):
    def bl(m):
        return ms[m]['best_loss']
    bm = sorted(list(ms.keys()), key=bl)
    if trim is not None and len(bm) > trim:
        bm = bm[:trim]
    return bm


models = load_models('models-sam')
best = get_best_models(models, 16)
for i in range(len(best)):
    model_name = best[i]
    model = models[model_name]
    children = 0
    for n, s in models.items():
        if s.get('parent', '?') == model_name:
            children += 1
    print('%3d: (vl: %.5f, va: %.5f, epochs: %3d, params: %4d, children: %d) %s <- %s' %
          (i + 1, model.get('best_loss', None), model.get('best_acc', 0), model.get('epoch', '0'), model.get('param_count', 0), children, model_name, model.get('parent', '???')))
