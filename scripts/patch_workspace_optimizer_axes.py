"""Replace only explicit W&B history-step axes, preserving epoch axes."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path


def build(spec):
    result = copy.deepcopy(spec)
    changed = []

    def visit(value, path=''):
        if isinstance(value, dict):
            if value.get('xAxis') == '_step':
                value['xAxis'] = 'train/step'
                changed.append(path + '/xAxis')
            for key, child in value.items():
                visit(child, path + '/' + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + '/' + str(index))

    visit(result)
    return result, changed


def main():
    import wandb
    from wandb_workspaces.workspaces import internal
    from wandb_workspaces.workspaces.internal import gql

    view = '3fvncok3gjh'
    before = internal.get_view_dict('peter-sk-sdu', 'DFM5', view)
    after, changed = build(json.loads(before['spec']))
    if not changed:
        print('No _step axes remain.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path('logs/wandb_workspace_specs') / f'{view}-optimizer-axes-{stamp}'
    out.mkdir(parents=True)
    for name, value in [('before', before), ('intended', after), ('changes', changed)]:
        (out / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
    if internal.get_view_dict('peter-sk-sdu', 'DFM5', view) != before:
        raise RuntimeError('Concurrent workspace edit; refusing overwrite')
    mutation = gql('''mutation UpdateExistingView($id: ID!, $spec: String!) {
        upsertView(input: {id: $id, spec: $spec}) {view {id name} inserted}
    }''')
    wandb.Api().client.execute(mutation, {'id': before['id'], 'spec': json.dumps(after)})
    actual = internal.get_view_dict('peter-sk-sdu', 'DFM5', view)
    (out / 'after.json').write_text(json.dumps(actual, indent=2) + '\n')
    assert actual['id'] == before['id'] and json.loads(actual['spec']) == after
    print(f'Verified {len(changed)} axis replacements. Backup: {out}')


if __name__ == '__main__':
    main()
