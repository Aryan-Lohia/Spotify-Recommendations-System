from pathlib import Path
import scipy
import pandas as pd


def load_user_artists(file: Path) -> scipy.sparse.csr_matrix:
    user_artists = pd.read_csv(file, sep="\t")
    user_artists.set_index(["userID", "artistID"], inplace=True)
    coo = scipy.sparse.coo_matrix((
        user_artists.weight.astype(float),
        (
            user_artists.index.get_level_values(0),
            user_artists.index.get_level_values(1)
        )
    ))
    return coo.tocsr()

class ArtistRetriever:
    def __init__(self):
        self._artists_df=None

    def get_artist_name_from_id(self,artist_id:int)->str:
        """
        Return the artist name from artist id
        :param artist_id:
        :return: String
        """
        return self._artists_df.loc[artist_id,"name"]

    def load_artists(self,artists_file:Path)->None:
        """
        Load the artist's file and stores it as a Pandas Dataframe in
        a private attribute
        :param artists_file:
        :return: None
        """
        artists_df=pd.read_csv(artists_file,sep="\t")
        artists_df.set_index("id",inplace=True)
        self._artists_df=artists_df

# if __name__ == "__main__":
    # user_artists_matrix = load_user_artists(
    #     Path("dataset/user_artists.dat")
    # )
    # print(user_artists_matrix)
    #
    # artist_obj=ArtistRetriever()
    # artist_obj.load_artists(Path("dataset/artists.dat"))
    # print(artist_obj.get_artist_name_from_id(1))