import pandas as pd
from definitions import DATA_DIR


def get_packs() -> pd.DataFrame:
    table_nek = pd.read_csv(DATA_DIR.joinpath('table_nek_2020_01.csv'))
    pack_counts = table_nek.groupby('path').count().pack_id.reset_index()
    pack_counts['path'] = pack_counts['path'].apply(lambda x: "/".join(x.split('/')[-2:]))
    pack_counts.rename(columns={'pack_id': 'packs_count'}, inplace=True)
    return pack_counts


if __name__ == '__main__':
    pass
    # print(get_packs())
