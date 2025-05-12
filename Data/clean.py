import pandas as pd
import zipfile
import os

def load_csv_from_zip(zip_path: str, csv_name: str) -> pd.DataFrame:
    """
    Abre o zip em zip_path, procura por um arquivo que termine em csv_name
    e retorna um DataFrame lido diretamente do zip.
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP não encontrado em: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as z:
        # procura o arquivo exato dentro do zip
        match = [n for n in z.namelist() if n.endswith(csv_name)]
        if not match:
            raise FileNotFoundError(f"'{csv_name}' não encontrado dentro de {zip_path}")
        csv_path = match[0]
        with z.open(csv_path) as f:
            df = pd.read_csv(f)
    print(f"Carregado '{csv_path}' do ZIP ({zip_path}), shape = {df.shape}")
    return df

def clean_ufc_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # DROP colunas não necessárias
    for c in ['event', 'origin_fight_url']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # identifica colunas numéricas (tudo que começa com fighter1_ ou fighter2_ exceto Stance)
    stats = [c for c in df.columns if (c.startswith('fighter1_') or c.startswith('fighter2_'))]
    num_cols = [c for c in stats if not c.endswith('Stance')]

    # tipos e medianas
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # one-hot encode de stances
    stance_cols = [c for c in stats if c.endswith('Stance')]
    for col in stance_cols:
        df[col] = df[col].fillna('Unknown').str.strip().str.capitalize()
    df = pd.get_dummies(df, columns=stance_cols, prefix=stance_cols)

    # colunas de diferença (fighter1_X – fighter2_X)
    base_stats = set(c.split('_',1)[1] for c in num_cols)
    for stat in base_stats:
        c1 = f'fighter1_{stat}'
        c2 = f'fighter2_{stat}'
        if c1 in df and c2 in df:
            df[f'diff_{stat}'] = df[c1] - df[c2]

    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    ZIP_PATH = "DataSet.zip"
    CSV_NAME = "clean_ufc_all_fights.csv"
    OUT_CSV = "ufc_fights_cleaned.csv"
    
# --- abre o ZIP e carrega o CSV interno ---
with zipfile.ZipFile("DataSet.zip", "r") as z:
    internal_csv = [f for f in z.namelist() if f.endswith("clean_ufc_all_fights.csv")][0]
    with z.open(internal_csv) as f:
        df = pd.read_csv(f)

    clean_df = clean_ufc_data(df)

    clean_df.to_csv("ufc_fights_cleaned.csv", index=False)
    print("Salvo como ufc_fights_cleaned.csv (formato CSV puro)")
