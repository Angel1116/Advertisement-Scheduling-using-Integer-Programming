import pandas as pd

genre_df = pd.read_csv('anime_type.csv')

genre_df = genre_df[["MAL_ID", "Genres"]]
genre_df = genre_df.rename(columns={"MAL_ID": "anime_id"})
genre_df = genre_df.set_index("anime_id")

genre_dummies = genre_df['Genres'].str.get_dummies(sep=', ')
final_df = genre_df.drop(columns=['Genres']).join(genre_dummies)

def set_genre(row):
    if row["Hentai"] == 1 or row["Ecchi"] == 1:
        return "Hentai"
    elif row["Kids"] == 1:
        return "Kids"
    elif row["Romance"] == 1 or row["Shoujo"] == 1:
        return "Romance"
    elif row["Sci-Fi"] == 1:
        return "Adventure"
    elif row["Action"] == 1 and row["Adventure"] == 1:
        return "Adventure"
    elif row["Action"] == 1 and row["Adventure"] == 0:
        return "Action"
    elif row["Action"] == 0 and row["Adventure"] == 1:
        return "Adventure"
    else:
        for i in ["Comedy","Music","Drama","Slice of Life","Fantasy","Dementia","Sports","Historical","Supernatural","School","Mystery","Game","Parody"]:
            if row[i] == 1:
                return i
                break

final_df["genre"] = final_df.apply(set_genre, axis=1)
final_df["genre"].fillna("others", inplace=True)

final_df = final_df[["genre"]]
final_df = pd.get_dummies(final_df, columns=["genre"], prefix="", prefix_sep="")

final_df .to_csv("genre_df.csv")