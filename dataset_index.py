"""Deterministic file grouping and user labels; no data is loaded here."""
from pathlib import Path

CROSSROOM_USERS = ('bin', 'changming', 'kailong', 'linqi', 'runtian', 'wenjin')
CROSSROOM_ACTIVITIES = ('doc', 'walk', 'sit', 'squat', 'wiping')


def user_name(file):
    """Normalize filename usernames, including Kailong/kailong."""
    return Path(file).name.split('_')[0].casefold()



def mat_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')
    files = sorted(root.glob('*.mat'))
    if not files:
        raise ValueError(f'No MAT files found in {root}')
    return [str(p) for p in files]


def dataset_groups(root, group_by='environment', environments=None, users=None):
    """Return (group name, absolute file list), preserving recording boundaries."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if environments is not None:
        names = list(environments)
    groups = [(name, mat_files(root / name)) for name in names]
    if users is not None:
        selected = {user.casefold() for user in users}
        groups = [(name, [f for f in files if user_name(f) in selected]) for name, files in groups]
        for name, files in groups:
            if not files:
                raise ValueError(f'No selected users in {root / name}')
    if not groups:
        raise ValueError(f'No dataset groups in {root}')
    if group_by == 'environment':
        return groups
    if group_by != 'user':
        raise ValueError(f'Unsupported group_by: {group_by}')
    grouped_users = {}
    for _, files in groups:
        for file in files:
            grouped_users.setdefault(user_name(file), []).append(file)
    return [(user, sorted(files)) for user, files in sorted(grouped_users.items())]


def user_label_map(root):
    """One shared, alphabetical user mapping across sibling dataset groups."""
    root = Path(root)
    files = sorted(root.parent.glob('*/*.mat'))
    if not files:
        files = [Path(p) for p in mat_files(root)]
    return {user: i for i, user in enumerate(sorted({user_name(p) for p in files}))}
